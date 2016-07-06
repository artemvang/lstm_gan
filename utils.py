import pickle
import argparse

SAVER_FILE = "model.ckpt"

def get_args():
    parser = argparse.ArgumentParser(description='Trainer for lstm-gan model.')
    parser.add_argument("disc_count", help="Count of dataset batches, \
        which will be train before generator train", type=int)
    parser.add_argument("gen_count", help="Count noise batches, \
        which will be train after discriminator train", type=int)
    parser.add_argument("--data_path", default="data", help="Dataset and vocabulary path", type=str)
    parser.add_argument("--batch_size", default=512, help="Batch_size", type=int)
    parser.add_argument("--hid_gen", default=512, help="Hidden size of lstm in generator", type=int)
    parser.add_argument("--hid_disc", default=512, help="Hidden size of lstm in discriminator", type=int)
    parser.add_argument("--dropout", default=0.2, help="Dropout", type=float)
    parser.add_argument("--grad_clip", default=1., help="Gradient clipping", type=float)
    parser.add_argument("--noise_size", default=32, help="Size of input noise into generator", type=int)
    parser.add_argument("-s", "--save_model", help="Save session", action="store_true")
    parser.add_argument("--load_model", help="Load session from file", action="store_true")
    parser.add_argument("--lr", default=1e-4, help="Learning rate for Adam optimizer", type=float)
    args = parser.parse_args()
    return args

def load_dataset(pickle_file):
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def load_dicts(index2word_file):    
    with open(index2word_file, 'rb') as f:
        index2word =  pickle.load(f)
    word2index = dict([(w,i) for i,w in enumerate(index2word)])
    return index2word, word2index

def iterate_over_dataset(dataset, batch_size):
    for i in range(0, int(dataset.shape[0]/batch_size)*batch_size, batch_size):
        batch = dataset[i:i+batch_size]
        yield batch