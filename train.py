import os
import glob

import tensorflow as tf
import neuralgym as ng

from data_from_fnames import DataFromFNames
from inpaint_model import InpaintCAModel



def multigpu_graph_def(model, FLAGS, data, gpu_id=0, loss_type='g'):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(FLAGS.batch_size)
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, FLAGS, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, FLAGS, summary=True, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


if __name__ == "__main__":
    # training data
    FLAGS = ng.Config('inpaint.yml')
    img_shapes = FLAGS.img_shapes
    with open(FLAGS.data_flist[FLAGS.dataset][0]) as f:
        fnames = f.read().splitlines()
        print(fnames[:5])
    if FLAGS.guided:
        fnames = [(fname, fname[:-4] + '_edge.jpg') for fname in fnames]
        img_shapes = [img_shapes, img_shapes]
    data = DataFromFNames(
        fnames, img_shapes, random_crop=FLAGS.random_crop,
        nthreads=FLAGS.num_cpus_per_job, adv_train = FLAGS.robust, 
    enqueue_size = FLAGS.batch_size)
    images = data.data_pipeline(FLAGS.batch_size)
    print(images.shape)
    
    # main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(FLAGS, images, summary=True)
    
    
    # validation images
    if FLAGS.val:
        with open(FLAGS.data_flist[FLAGS.dataset][1]) as f:
            val_fnames = f.read().splitlines()
        if FLAGS.guided:
            val_fnames = [
                (fname, fname[:-4] + '_edge.jpg') for fname in val_fnames]
        # progress monitor by visualizing static images
        for i in range(FLAGS.static_view_size):
            static_fnames = val_fnames[i:i+1]
            static_images = ng.data.DataFromFNames(
                static_fnames, img_shapes, nthreads=1,
                random_crop=FLAGS.random_crop).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                FLAGS, static_images, name='static_view/%d' % i)
    # training settings
    lr = tf.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.constant_initializer(1e-4))
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
    g_optimizer = d_optimizer
    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    # discriminator_training_callback = ng.callbacks.SecondaryTrainer(
    discriminator_training_callback = ng.callbacks.SecondaryMultiGPUTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=1,
        grads_summary=False,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'd'},
        log_progress=True,
    )
    # train generator with primary trainer
    # trainer = ng.train.Trainer(
    trainer = ng.train.MultiGPUTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=FLAGS.max_iters,
        graph_def=multigpu_graph_def,
        grads_summary=False,
        gradient_processor=None,
        graph_def_kwargs={
            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'g'},
        spe=FLAGS.train_spe,
        log_dir=FLAGS.log_dir,
        log_progress=True,
    )
    
    if FLAGS.robust:
        loss = losses['ae_loss'] - losses['g_loss']
        gradient = tf.gradients(loss , images)
        signed_grad = tf.sign(gradient)
        adv_x = images + FLAGS.epsilon*signed_grad
        adv_x = tf.clip_by_value(adv_x, 0, 255)
#         adv_x = tf.reverse(adv_x, [-1])          
        adv_x = tf.saturate_cast(adv_x, tf.uint8)
        data.images = images
        data.adv_op = adv_x
        data.has_init = True
        data.sess = trainer.context['sess']
        
    # add all callbacks
    trainer.add_callbacks([
        discriminator_training_callback,
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix=FLAGS.model_restore+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(FLAGS.train_spe, trainer.context['saver'], FLAGS.log_dir+'/snap'),
        ng.callbacks.SummaryWriter((FLAGS.val_psteps//1), trainer.context['summary_writer'], tf.summary.merge_all()),
    ])
    # launch training
    print("TRAINING STARTED")
    trainer.train()
