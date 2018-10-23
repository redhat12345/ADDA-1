import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
import h5py 
import os 
import scipy.io as io

'''
    dataset     training samples    test samples
    MNIST            60000             10000
    USPS             7291               2007
    SVHN             73257             26032        531131 for additional
'''

'''
digits datasets:
1. USPS 7291 training images and 2007 test images of size 16 x 16
2. MNIST 60000 training images and 10000 test images of size 28 x 28


experiments
1. USPS <-> MNIST randomly sampling 2000 images in MNIST to from the target data,
randomly sampling 1800 images in USPS to form the source data. rescale all images
to size 16 x 16, encoding images gray-scale pixel values,

'''
#============util function===============
def get_dataset(source,target):
    func = [None,None]
    for id,domain in enumerate([source,target]):
        if domain == "MNIST":
            func[id] = read_MNIST
        elif domain == "MNIST-M":
            func[id] = read_MNIST_M 
        elif domain == "USPS":
            func[id] = read_USPS
        elif domain == 'SVHN':
            func[id] = read_SVHN
    return func

def get_dataset_v2(name):
    if name == "MNIST":
        return read_MNIST
    elif name == "USPS":
        return read_USPS
    elif name == "MNIST-M":
        return read_MNIST_M
    elif name == "SVHN":
        return read_SVHN
    else:
        raise ValueError("can't find read data function for this dataset")

# ===========read_xxx=====================
def read_MNIST(batch_size,random_sample_train=2000,random_sample_test=2000):
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    #print(x_train.shape) # (60000,28,28)
    #print(y_train.shape) # (60000)
    #print(x_test.shape) # (10000,28,28)
    #print(y_test.shape) # (10000)
    if random_sample_train != None:
        idx = np.random.choice(np.arange(len(x_train)),random_sample_train)
        x_train = x_train[idx]
        y_train = y_train[idx]
    if random_sample_test != None:
        idx = np.random.choice(np.arange(len(x_test)),random_sample_test)
        x_test = x_test[idx]
        y_test = y_test[idx]

    # group to tf.data.dataset
    #print(np.max(x_train[1]))

    # create dataset object
    x_train = np.expand_dims(x_train,axis=-1)
    x_test = np.expand_dims(x_test,axis=-1)
    x_train = x_train * (1.0/127.5) - 1.0
    x_test = x_test * (1.0 /127.5) - 1.0
    train_dataset = tf.data.Dataset.from_tensor_slices({'image':x_train,'label':y_train})
    test_dataset = tf.data.Dataset.from_tensor_slices({'image':x_test,'label':y_test})
    # image preprocessing
    train_dataset = train_dataset.map(img_preprocessing)
    #train_dataset = train_dataset.apply(tf.contrib.data.shuffle_and_repeat(len(x_train))).batch(batch_size,drop_remainder=True)
    train_dataset =train_dataset.shuffle(len(x_train)).repeat().batch(batch_size,drop_remainder=True)
    #print(len(x_test))
    test_dataset = test_dataset.map(img_preprocessing).batch(len(x_test),drop_remainder=True)
    # create iter
    iter_tr = train_dataset.make_one_shot_iterator()
    iter_te = test_dataset.make_initializable_iterator()
    image_tr,label_tr = iter_tr.get_next()
    image_te,label_te = iter_te.get_next()

    train_size = len(x_train)
    test_size = len(x_test)
    return image_tr,label_tr,image_te,label_te,train_size,test_size,iter_te.initializer 

def read_USPS(batch_size,random_sample_train=1800,random_sample_test=2000):
    with h5py.File("./data/usps.h5",'r') as hf:
        train = hf.get('train')
        x_train = train.get('data')[:] # (7291,256)
        y_train = train.get('target')[:]# (7291,)
        test = hf.get('test')
        x_test = test.get('data')[:] # (2007,256)
        y_test = test.get('target')[:] # (2007,)
    if random_sample_train != None:
        idx = np.random.choice(np.arange(len(x_train)),random_sample_train)
        x_train = x_train[idx]
        y_train = y_train[idx]
    if random_sample_test != None:
        idx = np.random.choice(np.arange(len(x_test)),random_sample_test)
        x_test = x_test[idx]
        y_test = y_test[idx]

    x_train = np.reshape(x_train,[len(x_train),16,16,1])
    x_test = np.reshape(x_test,[len(x_test),16,16,1])
    x_train = (x_train -0.5) / 0.5
    x_test = (x_test-0.5)/0.5
    
    #show_grid(x_train,y_train,show_label=True)
    # create dataset object
    train_dataset = tf.data.Dataset.from_tensor_slices({'image':x_train,'label':y_train})
    test_dataset = tf.data.Dataset.from_tensor_slices({'image':x_test,'label':y_test})
    # image preprocessing
    train_dataset = train_dataset.map(img_preprocessing)
    #train_dataset = train_dataset.apply(tf.contrib.data.shuffle_and_repeat(len(x_train))).batch(batch_size,drop_remainder=True)
    train_dataset =train_dataset.shuffle(len(x_train)).repeat().batch(batch_size,drop_remainder=True)
    test_dataset = test_dataset.map(img_preprocessing).batch(len(x_test),drop_remainder=True)
    # create iter
    iter_tr = train_dataset.make_one_shot_iterator()
    iter_te = test_dataset.make_initializable_iterator()
    image_tr,label_tr = iter_tr.get_next()
    image_te,label_te = iter_te.get_next()

    train_size = len(x_train)
    test_size = len(x_test)
    return image_tr,label_tr,image_te,label_te,train_size,test_size,iter_te.initializer

def read_MNIST_M(batch_size,random_sample_train=2000,random_sample_test=None):
    root_path = "./data/mnist_m"
    train_dir = os.path.join(root_path,"mnist_m_train")
    test_dir = os.path.join(root_path,"mnist_m_test")
    train_label_path = os.path.join(root_path,'mnist_m_train_labels.txt')
    test_label_path = os.path.join(root_path,'mnist_m_test_labels.txt')
    # first read label file to get [filename,label] list.
    # training images
    f_tr = open(train_label_path)
    records_tr = np.array(f_tr.readlines())
    f_tr.close()
    if random_sample_train !=None:
        idx = np.random.choice(np.arange(len(records_tr)),random_sample_train)
        records_tr = records_tr[idx]

    # testing images
    f_te= open(test_label_path)
    records_te = np.array(f_te.readlines())
    f_te.close()
    if random_sample_test != None:
        idx = np.random.choice(np.arange(len(records_te)),random_sample_train)
        records_te = records_te[idx]

    # change dicts
    records_tr_ = {"image":[],"label":[]}
    for i in records_tr:
        item = i.strip().split()
        item[0] = os.path.join(train_dir,item[0])
        item[1] = int(item[1])
        records_tr_['image'].append(item[0])
        records_tr_['label'].append(item[1])
    
    records_te_ = {'image':[],"label":[]}
    for i in records_te:
        item = i.strip().split()
        item[0] = os.path.join(test_dir,item[0])
        item[1] = int(item[1])
        records_te_['image'].append(item[0])
        records_te_['label'].append(item[1])

    # create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(records_tr_)
    test_dataset = tf.data.Dataset.from_tensor_slices(records_te_)

    # use dataset.map to read image from file
    train_dataset = train_dataset.map(img_preprocessing_mnist_m)
    test_dataset = test_dataset.map(img_preprocessing_mnist_m)
    print(train_dataset.output_shapes)
    print(train_dataset.output_types)

    train_dataset = train_dataset.apply(tf.contrib.data.shuffle_and_repeat(len(records_tr_['image']))).batch(batch_size,drop_remainder=True)
    iter_tr = train_dataset.make_one_shot_iterator()
    image_tr,label_tr = iter_tr.get_next()

    test_dataset = test_dataset.batch(len(records_te_['image']))
    iter_te = test_dataset.make_initializable_iterator()
    image_te,label_te = iter_te.get_next()

    train_size = len(records_tr_['image'])
    test_size = len(records_te_['image'])
    return image_tr,label_tr,image_te,label_te,train_size,test_size,iter_te.initializer  

def read_SVHN(batch_size,random_sample_train=None,random_sample_test=None):
    # first, get raw data
    train_SVHN_path = "./data/SVHN/train_32x32.mat"
    test_SVHN_path = "./data/SVHN/test_32x32.mat"
    train_set=io.loadmat(train_SVHN_path)
    test_set=io.loadmat(test_SVHN_path)
    x_train = train_set['X']
    y_train = np.reshape(train_set['y'],(train_set['y'].shape[0]))
    x_test = test_set['X']
    y_test = np.reshape(test_set['y'],(test_set['y'].shape[0]))
    for idx,i in enumerate(y_train):
        if i == 10:
            y_train[idx] = 0
    for idx,i in enumerate(y_test):
        if i == 10:
            y_test[idx] = 0
    
    # process raw data
    x_train = np.transpose(x_train,[3,0,1,2])
    x_test = np.transpose(x_test,[3,0,1,2])

    # do random choice
    if random_sample_train!=None:
        idx = np.random.choice(np.arange(x_train.shape[0]),random_sample_train)
        x_train = x_train[idx]
        y_train = y_train[idx]
    if random_sample_test!=None:
        idx = np.random.choice(np.arange(x_test.shape[0]),random_sample_test)
        x_test = x_test[idx]
        y_test = y_test[idx]
    #print(type(y_test[0]))

    # group to tensorflow api
    train_dataset = tf.data.Dataset.from_tensor_slices({'image':x_train,'label':y_train})
    test_dataset = tf.data.Dataset.from_tensor_slices({'image':x_test,'label':y_test})
    # image process use map()
    train_dataset = train_dataset.map(img_preprocessing)
    #train_dataset = train_dataset.apply(tf.contrib.data.shuffle_and_repeat(len(x_train))).batch(batch_size,drop_remainder=True)
    train_dataset =train_dataset.shuffle(len(x_train)).repeat().batch(batch_size,drop_remainder=True)
    test_dataset = test_dataset.map(img_preprocessing).batch(len(x_test),drop_remainder=True)
    # create iter
    iter_tr = train_dataset.make_one_shot_iterator()
    iter_te = test_dataset.make_initializable_iterator()
    image_tr,label_tr = iter_tr.get_next()
    image_te,label_te = iter_te.get_next()
    train_size = len(x_train)
    test_size = len(x_test)
    return image_tr,label_tr,image_te,label_te,train_size,test_size,iter_te.initializer

# ==============img preprocess==============
def img_preprocessing_mnist_m(record):
    image = tf.image.decode_image(tf.read_file(record['image']))
    image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,[28,28])
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image,tf.float32)
    label = record['label']
    label = tf.cast(label,tf.int32)
    return image,label

def img_preprocessing(record):
    img = record['image']
    label = record['label']
    if img.shape[2] != 1:
        img = tf.image.rgb_to_grayscale(img)
    if img.shape[0] != 28:
        img = tf.image.resize_images(img,[28,28])
    #img = tf.image.per_image_standardization(img)
    img = tf.cast(img,tf.float32)
    label = tf.cast(label,tf.int32) 
    return img,label
# ==============image preprocess end==========================
def show_grid(images,labels,shape=[2,5],show_label=False):
    fig = plt.figure("images")
    if show_label:
        pad = 0.3
    else:
        pad = 0.05
    grid = ImageGrid(fig,111,nrows_ncols=shape,axes_pad=pad)
    
    size = shape[0] * shape[1]
    
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i],cmap=plt.cm.gray)
        if show_label:
            grid[i].set_title(labels[i])
    plt.show()

def show_grid_v2(fig,images,labels,shape=[2,5]):
    
    size = shape[0]*shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            id = i*shape[1]+j
            ax = fig.add_subplot(shape[0],shape[1],id+1)
            #print(images[i].shape)
            if images.shape[3] == 1:
                ax.imshow(images[id,:,:,0],cmap=plt.cm.gray)
            else:
                ax.imshow(images[id,:,:,:])
            ax.axis('off')
            ax.set_title(labels[id])
            
def test(type):
    batch_size = 10
    if type == "USPS":
        Xtr,ytr,Xte,yte,tr_size,te_size,_ = read_USPS(batch_size,random_sample_train=200)
    elif type == "MNIST":
        Xtr,ytr,Xte,yte,tr_size,te_size,_ = read_MNIST(batch_size,random_sample_train=200)
    elif type == "MNIST-M":
        Xtr,ytr,Xte,yte,tr_size,te_size,_ = read_MNIST_M(batch_size,random_sample_train=None)
    elif type == "SVHN":
        Xtr,ytr,Xte,yte,tr_size,te_size,_ = read_SVHN(batch_size,random_sample_train=None,random_sample_test=None)

    print(Xtr.shape)

    sess = tf.Session()
    batch_num = tr_size/batch_size
    print("batch_num:%d"%batch_num)
    plt.ion()
    fig = plt.figure("images")
    for i in range(int(batch_num)):
        print("batch_index:%d"%i)
        Xtr_,ytr_ = sess.run([Xtr,ytr])
        print(ytr_)
        print(np.max(Xtr_))
        print(np.min(Xtr_)) 
        show_grid_v2(fig,Xtr_,ytr_)
        plt.pause(0.01)
        plt.clf() # clear all the ax in the figures. Note that plt.cla is clear the axes now
    plt.ioff()
    #plt.show()
    plt.close()

if __name__ == "__main__":
    #stack_dataset_batch()
    test("USPS")
    #read_MNIST_M(10)
    #read_SVHN(19)