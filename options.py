import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=1200, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=2,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'dehaze', 'derain'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='/data2/liao/projects/All-in-One/PromptIR-main/data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/data2/liao/projects/All-in-One/PromptIR-main/data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/data2/liao/projects/All-in-One/PromptIR-main/data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/data2/liao/projects/All-in-One/PromptIR-main/data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--gopro_dir', type=str, default='/data2/liao/projects/All-in-One/PromptIR-main/data/Train/Deblur/',
                    help='where clean images of denoising saves.')
parser.add_argument('--enhance_dir', type=str, default='/data2/liao/projects/All-in-One/PromptIR-main/data/Train/Enhance/',
                    help='where clean images of denoising saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default=None,help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 4,help = "Number of GPUs to use for training")

options = parser.parse_args()

