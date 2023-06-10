import argparse

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate to be used")
    parser.add_argument("--train_dataset_dir", type=str, default='',
                        help="Path to the directory containing the train dataset.")
    parser.add_argument("--val_dataset_dir", type=str, default='',
                        help="Path to the directory containing the train dataset.")
    parser.add_argument("--dataset_name", type=str, default='',
                        help="Name of the dataset")
    parser.add_argument("--checkpoint_filepath", type=str, default=f'./checkpoints',
                        help="Path where checkpoints will be stored")
    parser.add_argument("--restore_from", type=str, default='',
                        help="Where restore model parameters from.")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="start epoch")
    parser.add_argument("--epochs", type=int, default=100,
                        help="total number of epochs")
    parser.add_argument("--iteration_per_epoch", type=int, default=4000,
                        help="number of iterations per epoch")
    parser.add_argument("--val_iteration_per_epoch", type=int, default=600,
                        help="number of iterations per epoch for validation set")
    parser.add_argument("--ex_name", type=str, default='ex1',
                        help="name of the experiment")    
    parser.add_argument("--input_shape", type=str, default='224,224')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--optimizer_name", type=str, default='Adam',
                        help="Name of the optimizer")

    return parser.parse_args()