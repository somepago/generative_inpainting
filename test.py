import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

parser.add_argument('--image_dir', default='', type=str,
                    help='The filename of image to be completed.')

parser.add_argument('--image_filelist', default='', type=str,
                    help='The filename of image to be completed.')

parser.add_argument("--epsilon", type=int, default=8)


parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')



if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
#     image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)
    mask = mask/255
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

#     assert image.shape == mask.shape

    h, w, _ = mask.shape
    grid = 8
#     image = image[:h//grid*grid, :w//grid*grid, :]
    image_tmp = np.expand_dims(mask, 0)
    mask = mask[:h//grid*grid, :w//grid*grid, 0:1]
#     print('Shape of image: {}'.format(image.shape))

    mask = np.expand_dims(mask, 0)
#     input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=sess_config)
        
    image = tf.placeholder(dtype= tf.float32 ,shape=image_tmp.shape)
    mask = tf.constant(mask, dtype=tf.float32)
    g_vars, d_vars, losses, output = model.build_graph_with_losses_adv(FLAGS, image,mask, training=False)
#         output = model.build_server_graph(FLAGS, input_image)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')    

    loss = losses['ae_loss'] - losses['g_loss']
#     loss = losses['g_loss']

    gradient = tf.gradients(loss , image)
    signed_grad = tf.sign(gradient)
    adv_x = image + args.epsilon *signed_grad  
    adv_x = tf.clip_by_value(adv_x, 0, 255)
    
    image_op = tf.placeholder(dtype= tf.float32 ,shape=image_tmp.shape)
    psnr = tf.image.psnr(image, image_op, max_val=255)
    ssim = tf.image.ssim(image, image_op, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    l1_err = tf.reduce_mean(tf.math.abs((image-image_op)/255, name=None))
    
    output_path = os.path.join(args.output,str(args.epsilon))
    
    if not os.path.exists(output_path):
                os.makedirs(output_path)

    with open(os.path.join(output_path, "meta_list.txt"), "a") as meta_file:
                meta_file.write("PSNR_pred" + "\t")
                meta_file.write("PSNR_attacked" + "\t")
                meta_file.write("L1_loss_pred"+ "\t")
                meta_file.write("L1_loss_attacked" + "\t")
                meta_file.write("SSIM_pred"+ "\t")
                meta_file.write("SSIM_attacked" + "\t") 
                meta_file.write("\n")
    count = 0
    metrics = []
    with open(args.image_filelist, 'r') as fh:
        
        for image_name in fh.readlines():
            count+=1
            if count> 1000:
                break
            img_path = os.path.join(args.image_dir,image_name).strip() + '.jpg'
            image_array = cv2.imread(img_path)
            image_array = image_array[:h//grid*grid, :w//grid*grid, :]
            image_array = np.expand_dims(image_array, 0)           
            
            losses_clean, result_clean = sess.run([losses, output], feed_dict={image:image_array})
            adv_x_array = sess.run(adv_x,feed_dict={image:result_clean[:,:,:,::-1]})[0]
            losses_val, attacked_result = sess.run([losses, output], feed_dict={image:adv_x_array})
#             import pdb;pdb.set_trace()
            
            psnr_true = sess.run([psnr], feed_dict={image:image_array,image_op:result_clean[:,:, :, ::-1]} )[0][0]
            psnr_attacked = sess.run([psnr], feed_dict={image:image_array,image_op:attacked_result[:,:, :, ::-1]})[0][0]
            l1_true = sess.run([l1_err], feed_dict={image:image_array,image_op:result_clean[:,:, :, ::-1]})[0]
            l1_attacked = sess.run([l1_err], feed_dict={image:image_array,image_op:attacked_result[:,:, :, ::-1]})[0]
            ssim_true = sess.run([ssim], feed_dict={image:image_array,image_op:result_clean[:,:, :, ::-1]} )[0][0]
            ssim_attacked = sess.run([ssim], feed_dict={image:image_array,image_op:attacked_result[:,:, :, ::-1]})[0][0]
    
            row = np.concatenate(( image_array[0], result_clean[0][:, :, ::-1],adv_x_array[0],attacked_result[0][:, :, ::-1]), axis=1)
            
            metrics.append([psnr_true,psnr_attacked,l1_true,
                               l1_attacked,ssim_true,ssim_attacked])
            mean_metrics = np.mean(metrics, axis=0)

            if count%200==0:
                with open(os.path.join(output_path, "sum_meta.txt"), "a") as meta_file:
                    meta_file.write("PSNR_pred: {:.8f}".format(mean_metrics[2]) + "\t")
                    meta_file.write("PSNR_attacked: {:.8f}".format(mean_metrics[3]) + "\t")
                    meta_file.write("L1_loss_pred: {:.8f}".format(mean_metrics[4]) + "\t")
                    meta_file.write("L1_loss_attacked: {:.8f}".format(mean_metrics[5]) + "\t")
                    meta_file.write("SSIM_pred: {:.8f}".format(mean_metrics[6]) + "\t")
                    meta_file.write("SSIM_attacked: {:.8f}".format(mean_metrics[7]) + "\n")
                metrics = []
            
            cv2.imwrite(os.path.join(output_path,image_name + "_op.png"),row)
            with open(os.path.join(output_path, "meta_list.txt"), "a") as meta_file:
                            meta_file.write("{:.5f}".format(psnr_true) + "\t")
                            meta_file.write("{:.5f}".format(psnr_attacked) + "\t")
                            meta_file.write("{:.5f}".format(l1_true) + "\t")
                            meta_file.write("{:.5f}".format(l1_attacked) + "\t")
                            meta_file.write("{:.5f}".format(ssim_true) + "\t")
                            meta_file.write("{:.5f}".format(ssim_attacked) + "\t") 
                            meta_file.write("\n")
                            
                        
            print(count)


