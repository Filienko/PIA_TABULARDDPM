import numpy as np

import torch
from rich.progress import track
import fire
import logging
from rich.logging import RichHandler
from pytorch_lightning import seed_everything
import components
from typing import Type, Dict
from itertools import chain
from model import UNet
from dataset_utils import load_member_data
from torchmetrics.classification import BinaryAUROC, BinaryROC
import matplotlib.pyplot as plt


def get_FLAGS():

    def FLAGS(x): return x
    FLAGS.T = 1000
    FLAGS.ch = 128
    FLAGS.ch_mult = [1, 2, 2, 2]
    FLAGS.attn = [1]
    FLAGS.num_res_blocks = 2
    FLAGS.dropout = 0.1
    FLAGS.beta_1 = 0.0001
    FLAGS.beta_T = 0.02

    return FLAGS


def get_model(ckpt, WA=True):
    FLAGS = get_FLAGS()
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(ckpt, map_location='cpu')

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('module.'):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model


class EpsGetter(components.EpsGetter):
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        t = torch.ones([xt.shape[0]], device=xt.device).long() * t
        return self.model(xt, t=t)


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "SecMI": components.SecMIAttacker,
    "PIA": components.PIA,
    "naive": components.NaiveAttacker,
    "PIAN": components.PIAN,
}


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_distance_distribution(members, nonmembers, interval, save_path=None):
    """
    Create a three-panel visualization of distance distributions with increasing zoom levels.
    
    Args:
        members: List of distance values for members
        nonmembers: List of distance values for non-members
        interval: The interval value for the filename
        save_path: Path to save the plot and statistics
    """
    # Convert to numpy arrays for easier manipulation
    member_distances = np.concatenate([m.cpu().numpy() for m in members])
    nonmember_distances = np.concatenate([nm.cpu().numpy() for nm in nonmembers])
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), height_ratios=[1, 1, 1.5])
    
    # Plot 1: Full distribution
    bins = np.linspace(
        min(np.min(member_distances), np.min(nonmember_distances)),
        max(np.max(member_distances), np.max(nonmember_distances)),
        30
    )
    ax1.hist(nonmember_distances, bins=bins, alpha=0.6, 
             label=f'Non-member (n={len(nonmember_distances)})', 
             color='blue')
    ax1.hist(member_distances, bins=bins, alpha=0.6,
             label=f'Member (n={len(member_distances)})', 
             color='red')
    
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Count')
    ax1.set_title('Full Distribution of Distances')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # First zoom: Find the densest region
    all_data = np.concatenate([member_distances, nonmember_distances])
    Q1 = np.percentile(all_data, 25)
    Q3 = np.percentile(all_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 0.5 * IQR
    upper_bound = Q3 + 0.5 * IQR

    # Filter data for first zoom
    nonmember_filtered = nonmember_distances[
        (nonmember_distances >= lower_bound) & 
        (nonmember_distances <= upper_bound)
    ]
    member_filtered = member_distances[
        (member_distances >= lower_bound) & 
        (member_distances <= upper_bound)
    ]

    # Plot 2: First zoom level
    detailed_bins = np.linspace(lower_bound, upper_bound, 50)
    ax2.hist(nonmember_filtered, bins=detailed_bins, alpha=0.6,
             label=f'Non-member (n={len(nonmember_filtered)})', 
             color='blue')
    ax2.hist(member_filtered, bins=detailed_bins, alpha=0.6,
             label=f'Member (n={len(member_filtered)})', 
             color='red')
    
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Count')
    ax2.set_title('First Zoom Level\n'
                 f'(Range: {lower_bound:.2f} to {upper_bound:.2f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Second zoom: Find the even denser region
    filtered_data = np.concatenate([member_filtered, nonmember_filtered])
    Q1_filtered = np.percentile(filtered_data, 25)
    Q3_filtered = np.percentile(filtered_data, 75)
    IQR_filtered = Q3_filtered - Q1_filtered
    lower_bound_filtered = Q1_filtered - 0.25 * IQR_filtered
    upper_bound_filtered = Q3_filtered + 0.25 * IQR_filtered

    # Filter data for second zoom
    nonmember_filtered_2 = nonmember_filtered[
        (nonmember_filtered >= lower_bound_filtered) & 
        (nonmember_filtered <= upper_bound_filtered)
    ]
    member_filtered_2 = member_filtered[
        (member_filtered >= lower_bound_filtered) & 
        (member_filtered <= upper_bound_filtered)
    ]

    # Plot 3: Second zoom level with very fine bins
    very_detailed_bins = np.linspace(lower_bound_filtered, upper_bound_filtered, 100)
    ax3.hist(nonmember_filtered_2, bins=very_detailed_bins, alpha=0.6,
             label=f'Non-member (n={len(nonmember_filtered_2)})', 
             color='blue')
    ax3.hist(member_filtered_2, bins=very_detailed_bins, alpha=0.6,
             label=f'Member (n={len(member_filtered_2)})', 
             color='red')
    
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Count')
    ax3.set_title('Second Zoom Level (Finest Detail)\n'
                 f'(Range: {lower_bound_filtered:.2f} to {upper_bound_filtered:.2f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add statistical information for the finest zoom level
    stats_text = (
        f'Dense Region Stats:\n'
        f'Non-member:\n'
        f'  Mean: {np.mean(nonmember_filtered_2):.3f}\n'
        f'  Std: {np.std(nonmember_filtered_2):.3f}\n'
        f'Member:\n'
        f'  Mean: {np.mean(member_filtered_2):.3f}\n'
        f'  Std: {np.std(member_filtered_2):.3f}'
    )
    ax3.text(0.02, 0.98, stats_text, 
             transform=ax3.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/distance_distribution_T={interval}.png", 
                   dpi=300, bbox_inches='tight')
        
        # Save detailed statistics
        stats_summary = pd.DataFrame({
            'Metric': ['Full Mean', 'Full Std', 
                      'First Zoom Mean', 'First Zoom Std',
                      'Second Zoom Mean', 'Second Zoom Std',
                      'Points in Densest Region', 'Total Points'],
            'Member': [
                np.mean(member_distances),
                np.std(member_distances),
                np.mean(member_filtered),
                np.std(member_filtered),
                np.mean(member_filtered_2),
                np.std(member_filtered_2),
                len(member_filtered_2),
                len(member_distances)
            ],
            'Non-member': [
                np.mean(nonmember_distances),
                np.std(nonmember_distances),
                np.mean(nonmember_filtered),
                np.std(nonmember_filtered),
                np.mean(nonmember_filtered_2),
                np.std(nonmember_filtered_2),
                len(nonmember_filtered_2),
                len(nonmember_distances)
            ]
        })
        stats_summary.to_csv(f"{save_path}/detailed_statistics_T={interval}.csv", 
                           index=False)
    
    plt.close()


@torch.no_grad()
def main(checkpoint,
         dataset,
         attacker_name="naive",
         attack_num=30, interval=10,
         seed=0):
    seed_everything(seed)

    FLAGS = get_FLAGS()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info("loading model...")
    model = get_model(checkpoint, WA = True).to(DEVICE)
    model.eval()

    logger.info("loading dataset...")
    if dataset == 'cifar10':
        _, _, train_loader, test_loader = load_member_data(dataset_name='cifar10', batch_size=64,
                                                           shuffle=False, randaugment=False)
    if dataset == 'TINY-IN':
        _, _, train_loader, test_loader = load_member_data(dataset_name='TINY-IN', batch_size=64,
                                                           shuffle=False, randaugment=False)

    attacker = attackers[attacker_name](
        torch.from_numpy(np.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T)).to(DEVICE), interval, attack_num, EpsGetter(model), lambda x: x * 2 - 1, lp=4)

    logger.info("attack start...")
    members, nonmembers = [], []
    for member, nonmember in track(zip(train_loader, chain(*([test_loader]))), total=len(test_loader)):
        member, nonmember = member[0].to(DEVICE), nonmember[0].to(DEVICE)

        members.append(attacker(member))
        nonmembers.append(attacker(nonmember))

        members = [torch.cat(members, dim=-1)]
        nonmembers = [torch.cat(nonmembers, dim=-1)]

    member = members[0]
    nonmember = nonmembers[0]
    print(members.shape, nonmembers.shape)    
    print(member.shape, nonmember.shape)    
    print("member sample", member)    
    print("nonmember sample", nonmember)    

    auroc = [BinaryAUROC().cuda()(torch.cat([member[i] / max([member[i].max().item(), nonmember[i].max().item()]), nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()])]), torch.cat([torch.zeros(member.shape[1]).long(), torch.ones(nonmember.shape[1]).long()]).cuda()).item() for i in range(member.shape[0])]
    tpr_fpr = [BinaryROC().cuda()(torch.cat([1 - nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()]), 1 - member[i] / max([member[i].max().item(), nonmember[i].max().item()])]), torch.cat([torch.zeros(member.shape[1]).long(), torch.ones(nonmember.shape[1]).long()]).cuda()) for i in range(member.shape[0])]
    tpr_fpr_1 = [i[1][(i[0] < 0.01).sum() - 1].item() for i in tpr_fpr]
    cp_auroc = auroc[:]
    cp_auroc.sort(reverse=True)
    cp_tpr_fpr_1 = tpr_fpr_1[:]
    cp_tpr_fpr_1.sort(reverse=True)
    print('auc', auroc)
    print('tpr @ 1% fpr', cp_tpr_fpr_1)


if __name__ == '__main__':
    main(checkpoint="ckpt_cifar10.pt",
         dataset="cifar10",
         attacker_name="PIA",
         attack_num=30, interval=10,
         seed=0)