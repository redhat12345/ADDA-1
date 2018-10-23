import tensorflow as tf 
import dataset 
import matplotlib.pyplot as plt 
import utils 
import adda 

 
def step1(source="MNIST",batch_size=64,epoch=10,lr=0.001,
            logdir="./Log/ADDA/source_network/best/MNIST/NOBN",
            training_size=None,testing_size=None,classes_num=10):
    data_func = dataset.get_dataset_v2(source)
    x_tr,y_tr,x_te,y_te,tr_size,te_size,te_init = data_func(batch_size,training_size,testing_size)
    print("Training size:{},Testing size:{}".format(tr_size,te_size))
    batch_num = int(tr_size / batch_size)

    nn = adda.ADDA(classes_num)
    # inference classification network
    fc1 = nn.s_encoder(x_tr)
    logits = nn.classifier(fc1)

    # build loss and create optimizer
    c_loss = nn.build_classify_loss(logits,y_tr)
    train_op = tf.train.AdamOptimizer(lr).minimize(c_loss)

    # build training accuracy with training batch
    tr_acc = nn.eval(logits,y_tr)
    # build testing accuracy with testing data
    logits_te = nn.classifier(nn.s_encoder(x_te,reuse=True),reuse=True)
    te_acc = nn.eval(logits_te,y_te)

    # build saver to save best epoch
    var_s_en = tf.trainable_variables(scope=nn.s_e)
    var_c = tf.trainable_variables(scope=nn.c)
    encoder_saver = tf.train.Saver(max_to_keep=3,var_list=var_s_en)
    classifier_saver = tf.train.Saver(max_to_keep=3,var_list=var_c)
    # keep the logdir is empty
    utils.fresh_dir(logdir)

    # create a list to record accuracy in every batch
    eval_acc = []
    best_acc = 0
    
    # start a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for j in range(batch_num):
                _,loss,tr_acc_ = sess.run([train_op,c_loss,tr_acc])
                if j % 500 == 0:
                    print("epoch:{},batch_id:{},loss:{:.4f},tr_acc:{:.4f}".format(i,j,loss,tr_acc_))

            sess.run(te_init)
            te_acc_ = sess.run(te_acc)
            eval_acc.append(te_acc_)
            if best_acc < te_acc_:
                best_acc = te_acc_
                encoder_saver.save(sess,logdir+"/encoder/encoder.ckpt")  
                classifier_saver.save(sess,logdir+"/classifier/classifier.ckpt")
            print("#+++++++++++++++++++++++++++++++++++#")
            print("epoch:{},test_accuracy:{:.4f},best_acc:{:.4f}".format(i,te_acc_,best_acc))
            print("#+++++++++++++++++++++++++++++++++++#")
        utils.plot_acc(eval_acc,threshold=0.97,name=source+" test accuracy")
        plt.show()

def step2(source,target,epoch,batch_size=64,
          g_lr=0.0001,d_lr=0.0001,
          source_dir='./Log/ADDA/source_network/best/MNIST/NOBN',
          logdir = './Log/ADDA/advermodel/best/MNIST2USPS/NOBN',
          classes_num=10,strn=None,sten=None,ttrn=None,tten=None):
    # prepare data
    data_func = dataset.get_dataset(source,target)
    print(data_func)

    s_x_tr,s_y_tr,s_x_te,s_y_te,s_tr_size,s_te_size,s_init = data_func[0](batch_size,strn,sten)
    t_x_tr,t_y_tr,t_x_te,t_y_te,t_tr_size,t_te_size,t_init = data_func[1](batch_size,ttrn,tten)
    print("dataset information:\n source: %s train_size: %d, test_size: %d \n target: %s train_size: %d, test_size: %d"%(source,s_tr_size,s_te_size,target,t_tr_size,t_te_size))

    # create graph
    nn = adda.ADDA(classes_num)
    # for source domain
    feat_s = nn.s_encoder(s_x_tr,reuse=False,trainable=False)
    logits_s = nn.classifier(feat_s,reuse=False,trainable=False)
    disc_s = nn.discriminator(feat_s,reuse=False)

    # for target domain
    feat_t = nn.t_encoder(t_x_tr,reuse=False)
    logits_t = nn.classifier(feat_t,reuse=True,trainable=False)
    disc_t = nn.discriminator(feat_t,reuse=True)
    
    # build inference for test accuracy
    feats_s_te = nn.s_encoder(s_x_te,reuse=True,trainable=False)
    logits_s_te = nn.classifier(feats_s_te,reuse=True,trainable=False)
    disc_s_te = nn.discriminator(feats_s_te,reuse=True,trainable=False)

    feats_t_te = nn.t_encoder(t_x_te,reuse=True,trainable=False)
    logits_t_te = nn.classifier(feats_t_te,reuse=True,trainable=False)
    disc_t_te = nn.discriminator(feats_t_te,reuse=True,trainable=False)

    # build loss
    g_loss,d_loss = nn.build_ad_loss(disc_s,disc_t)
    #g_loss,d_loss = nn.build_w_loss(disc_s,disc_t)

    # create optimizer for two task
    var_t_en = tf.trainable_variables(nn.t_e)
    optim_g = tf.train.AdamOptimizer(g_lr,beta1=0.5,beta2=0.999).minimize(g_loss,var_list=var_t_en)
    #optim_g = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(g_loss,var_list=var_t_en)

    var_d = tf.trainable_variables(nn.d)
    optim_d = tf.train.AdamOptimizer(d_lr,beta1=0.5,beta2=0.999).minimize(d_loss,var_list=var_d)
    #optim_d = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(d_loss,var_list=var_d)
    #clip_D = [var.assign(tf.clip_by_value(var,-0.01,0.01)) for var in var_d]

    # create acuuracy op with training batch
    acc_tr_s = nn.eval(logits_s,s_y_tr)
    acc_tr_t = nn.eval(logits_t,t_y_tr)
    acc_te_s = nn.eval(logits_s_te,s_y_te)
    acc_te_t = nn.eval(logits_t_te,t_y_te)

    # create source saver for restore s_encoder
    encoder_path = tf.train.latest_checkpoint(source_dir+"/encoder")
    classifier_path = tf.train.latest_checkpoint(source_dir+"/classifier")
    if encoder_path is None:
        raise ValueError("Don't exits in this dir")
    if classifier_path is None:
        raise ValueError("Don't exits in this dir")

    source_var = tf.contrib.framework.list_variables(encoder_path)

    var_s_g = tf.global_variables(scope=nn.s_e)
    var_c_g = tf.global_variables(scope=nn.c)
    var_t_g = tf.trainable_variables(scope=nn.t_e)
    # print("+++++++++++++++")
    # print("s_encoder:",len(var_s_g))
    # print(var_s_g)
    # print("t_encoder:",len(var_t_g))
    # print(var_t_g)
    # print("source s_encoder:",len(source_var))
    # print(source_var)
    # print("+++++++++++++++")
    encoder_saver = tf.train.Saver(var_list=var_s_g)
    classifier_saver = tf.train.Saver(var_list=var_c_g)
    dict_var={}
    #print(type(source_var[0][0]))
    #print(type(var_t_g[0].name))
    for i in source_var:
        for j in var_t_g:
            if i[0][1:] in j.name[1:]:
                dict_var[i[0]]=j 
    #print(dict_var)
    fine_turn_saver = tf.train.Saver(var_list = dict_var)
    #assert False 
    # create this model saver
    utils.fresh_dir(logdir)
    best_saver = tf.train.Saver(max_to_keep=3)


    # create a list to record accuracy
    eval_acc = []
    best_acc = 0
    merge = tf.summary.merge_all()

    # start a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # init t_e and d
        sess.run(tf.global_variables_initializer())
        # init s_e and c
        encoder_saver.restore(sess,encoder_path)
        classifier_saver.restore(sess,classifier_path)
        fine_turn_saver.restore(sess,encoder_path)
        print("model init successfully!")
        filewriter = tf.summary.FileWriter(logdir=logdir,graph=sess.graph)
        for i in range(epoch):
            _,d_loss_,_,g_loss_,merge_ = sess.run([optim_d,d_loss,optim_g,g_loss,merge])
            filewriter.add_summary(merge_,global_step=i)
            if i % 20 == 0:
                print("step:{},g_loss:{:.4f},d_loss:{:.4f}".format(i,g_loss_,d_loss_))
            
            if i%100 == 0 or i>(epoch-100):
                sess.run([s_init,t_init])
                s_acc,t_acc,sx,sfe,sl,tx,tfe,tl = sess.run([acc_te_s,acc_te_t,s_x_te,logits_s_te,s_y_te,t_x_te,logits_t_te,t_y_te])
                eval_acc.append(t_acc)
                if best_acc < t_acc:
                    best_acc = t_acc
                print("epoch: %d, source accuracy: %.4f, target accuracy: %.4f, best accuracy：%4f"%(i,s_acc,t_acc,best_acc))
                best_saver.save(sess,logdir+"/adda_model.ckpt")
        utils.plot_acc(eval_acc,threshold=0.766)
        plt.show()

def step3(source,target,batch_size=64,logdir="./Log/ADDA/advermodel/best/MNIST2USPS/NOBN",
         classes_num=10,strn=None,sten=None,ttrn=None,tten=None):
    # prepare data
    data_func = dataset.get_dataset(source,target)
    print(data_func)

    s_x_tr,s_y_tr,s_x_te,s_y_te,s_tr_size,s_te_size,s_init = data_func[0](batch_size,strn,sten)
    t_x_tr,t_y_tr,t_x_te,t_y_te,t_tr_size,t_te_size,t_init = data_func[1](batch_size,ttrn,tten)
    print("dataset information:\n source: %s train_size: %d, test_size: %d \n target: %s train_size: %d, test_size: %d"%(source,s_tr_size,s_te_size,target,t_tr_size,t_te_size))

    # create graph
    nn = adda.ADDA(classes_num)
    # for source domain
    feat_s = nn.s_encoder(s_x_te,reuse=False,trainable=False)
    logits_s = nn.classifier(feat_s,reuse=False,trainable=False)
    disc_s = nn.discriminator(feat_s,reuse=False,trainable=False)

    # for target domain
    feat_t = nn.t_encoder(t_x_te,reuse=False,trainable=False)
    logits_t = nn.classifier(feat_t,reuse=True,trainable=False)
    disc_t = nn.discriminator(feat_t,reuse=True,trainable=False)

    source_accuracy = nn.eval(logits_s,s_y_te)
    target_accuracy = nn.eval(logits_t,t_y_te)

    path = tf.train.latest_checkpoint(logdir)
    saver = tf.train.Saver(max_to_keep=3)

    if path is None:
        raise ValueError("Don't exits in this dir:%s"%path)
    with tf.Session() as sess:
        saver.restore(sess,path)
        sess.run([s_init,t_init])
        s_acc,t_acc,sx,sfe,sl,tx,tfe,tl = sess.run([source_accuracy,target_accuracy,s_x_te,logits_s,s_y_te,t_x_te,logits_t,t_y_te])
        print(s_acc,t_acc)
        utils.plot_tsne(sfe,sl,tfe,tl,200)
        utils.plot_tsne_orign(sx,sl,tx,tl,200)
    plt.show()

    