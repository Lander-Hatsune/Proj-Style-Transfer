import keras.backend as K
from keras.application import vgg16
'''
input_tensor
img_nrows
img_ncols
'''
base_image_path = '.'
style_image_path = '.'


model = vgg16.VGG16(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
def preprocess_image(image_path) :
    img = load_img(image_path, target_size = (img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = vgg16.preprocess_input(img)
    return img

mean_B = 103.939
mean_G = 116.779
mean_R = 123.68

def deprocess_image(x) :
    if K.image_dim_ordering() == 'th' :
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else :
        x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += mean_B
    x[:, :, 1] += mean_G
    x[:, :, 2] += mean_R
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

base_image = K.variable(preprocess_image(base_image_path))
style_image = K.variable(preprocess_image(style_image_path))

if K.image_dim_ordering() == 'th' :
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else :
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

input_tensor = K.concatenate([base_image, style_image, combination_image], axis = 0)

def gram_matrix(x) :
    assert K.ndim(x) == 3
    if K.image_dim_ordering() == 'th' :
        features = K.batch_flatten(x)
    else :
        features = K.batch_flatten(K.permute_dimentions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination) :
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.squares(S - C)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(base, combination) :
    return K.sum(K.square(combination - base))

def total_variation_loss(x) :
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'th' :
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

loss = K.variable(0.)

layer_features = outputs_dict['block4_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features, combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

for layer_name in feature_layers :
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)
outputs = [loss]

if type(grads) in {list, tuple} :
    outputs += grads
else :
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)
