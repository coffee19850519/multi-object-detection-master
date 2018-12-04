import os

train_task_id = '3T0736'
initial_epoch = 0
epoch_num = 50
lr = 1e-3
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 20
load_weights = True
lambda_inside_score_loss = 1.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 17
validation_split_ratio = 0.3
max_train_img_size = 736#int(train_task_id[-3:])
max_predict_img_size = 736#int(train_task_id[-3:])  # 2400
#assert max_train_img_size in [256, 384, 512, 640, 736], \
#    'max_train_img_size must in [256, 384, 512, 640, 736]'
#if max_train_img_size < 256:
#    batch_size = 2000
#elif max_train_img_size < 384:
#    batch_size = 1000
#elif max_train_img_size < 512:
#    batch_size = 500
#else:
batch_size = 5

steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

data_dir = r'C:\Users\LSC-110\Desktop\training_data'
origin_image_dir_name = r'images'
origin_txt_dir_name = r'annotations'
train_image_dir_name = r'images_%s' % train_task_id
train_label_dir_name = r'labels_%s' % train_task_id
show_gt_image_dir_name = r'show_gt_images_%s' % train_task_id
show_act_image_dir_name = r'show_act_images_%s' % train_task_id
gen_origin_img = True
draw_gt_quad = True
draw_act_quad = True
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]

locked_layers = True

if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')

saved_checkpoint_weights_file = r'saved_model\weights_%s.{epoch:03d}-{' \
                        r'val_loss:.3f}.h5' \
                     % train_task_id
saved_last_model_file = r'saved_model\east_model_%s.h5' % train_task_id
saved_last_weight_file = r'saved_model\east_weights_%s.h5' % train_task_id
load_model_weights_file = \
  r'C:\Users\LSC-110\Desktop\keras-AdvancedEAST-master\load_model\east_model_weights_3T736.h5'
  #'saved_model/east_model_weights_%s.h5'\
   #                             % train_task_id

pixel_threshold = 0.8
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True
