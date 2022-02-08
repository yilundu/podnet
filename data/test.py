import tensorflow as tf
from robonet.datasets.robonet_dataset import RoboNetDataset
from robonet.datasets import load_metadata

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    all_robonet = load_metadata("/data/vision/billf/scratch/yilundu/robonet/hdf5")
    database = all_robonet[all_robonet['adim']==4]
    data = RoboNetDataset(batch_size=16, dataset_files_or_metadata=database, hparams={'img_size': [1024, 1024], 'load_T': 2, 'target_adim':4, 'action_mismatch':1})
    images = data['images']
    real_image = sess.run(images)
    import pdb
    pdb.set_trace()
    print("here")
    assert False

