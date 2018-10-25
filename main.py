import argparse
import os
import train 
import tensorflow as tf 
os.environ['CUDA_VISIBLE_DEVICES']='1'
tf.device('/gpu:1')

def main():
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--epoch',type=int,default=20)
    args = parser.parse_args()
    if args.step == 1:
        train.step1(source="MNIST",epoch=args.epoch)
        return 
    elif args.step == 2:
        train.step2(source="MNIST",target="USPS",epoch=args.epoch)
        return 
    elif args.step == 3:
        train.step3("MNIST","USPS")
        return 

if __name__ == "__main__":
    main()