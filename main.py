import argparse
import train_ecg
import train_mi


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    return False

def run(args):
    print(args)
    epochs = args.epochs
    model_name = args.model
    smote = args.smote
    batch_size= args.batch_size
    transfer_path = args.transfer_path
    mi = args.mi
    lr = args.lr
    save_model_path=args.save_model_path
    saved_loader = args.saved_loader
    
    if mi:
        train_mi.run_mi(model_name, smote=smote, num_epochs=epochs, batch_size=batch_size, learning_rate=lr, transfer=transfer_path, save_path=save_model_path)
    else:
        train_ecg.run_ecg(model_name, smote=smote, num_epochs=epochs, batch_size=batch_size, learning_rate=lr, save_path=save_model_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--model', type=str, default="cnet")
    parser.add_argument('--smote', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--transfer_path', type=str, default='')
    parser.add_argument('--mi', type=str2bool, default=False)
    parser.add_argument('--save_model_path', type=str, default='')
    parser.add_argument('--saved_loader', type=str, default='')
    
    args = parser.parse_args()
    run(args)

main()