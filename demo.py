import sys
sys.path.append('/home/caiyaom/python_codes/')
sys.path.append('/root/python_codes/')
from GR_ResSCNet import GRegConvAE
import numpy as np



if __name__ == '__main__':
    # load img and gt
    from Toolbox.Preprocessing import Processor
    from sklearn.preprocessing import minmax_scale
    from sklearn.decomposition import PCA
    import time

    root = 'D:\Python\HSI_Files\\'
    # root = '/home/caiyaom/HSI_Files/'
    # root = '/root/python_codes/HSI_Files/'
    im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    # im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    # im_, gt_ = 'PaviaU', 'PaviaU_gt'
    # im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
    # im_, gt_ = 'Botswana', 'Botswana_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'
    # im_, gt_ = 'Houston', 'Houston_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    # for nb_comps in range(2, 31, 1):
    # for size in range(5, 31, 2):
    NEIGHBORING_SIZE = 13
    EPOCH = 100
    LEARNING_RATE = 0.0002
    REG_LAP = 0.001  # beta
    REG_LATENT = 100.  # alpha
    WEIGHT_DECAY = 0.001  # lambda
    SEED = None  # random seed
    nb_comps = 5
    VERBOSE_TIME = 10
    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    if im_ == 'PaviaU':
        img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]
        SEED = 33
        NEIGHBORING_SIZE = 17
        EPOCH = 1000
        LEARNING_RATE = 0.0002
        REG_LAP = 0.001  # beta
        REG_LATENT = 100.  # alpha
        WEIGHT_DECAY = 0.001  # lambda
        VERBOSE_TIME = 100
    if im_ == 'Indian_pines_corrected':
        img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]
        SEED = 133
        NEIGHBORING_SIZE = 13
        EPOCH = 100
        LEARNING_RATE = 0.0002
        REG_LAP = 0.001  # beta
        REG_LATENT = 100.  # alpha
        WEIGHT_DECAY = 0.001  # lambda
        VERBOSE_TIME = 10
    if im_ == 'Salinas_corrected':
        img, gt = img[0:140, 50:200, :], gt[0:140, 50:200]
        SEED = 123
    if im_ == 'SalinasA_corrected':
        SEED = 10
        NEIGHBORING_SIZE = 9
        EPOCH = 100
        LEARNING_RATE = 0.0002
        REG_LAP = 0.001  # beta
        REG_LATENT = 100.  # alpha
        WEIGHT_DECAY = 0.001  # lambda
        VERBOSE_TIME = 10
    if im_ == 'Houston':
        img, gt = img[:, 0:680, :], gt[:, 0:680]
        SEED = 133
        NEIGHBORING_SIZE = 9
        EPOCH = 50
        LEARNING_RATE = 0.0002
        REG_LAP = 0.001  # beta
        REG_LATENT = 100.  # alpha
        WEIGHT_DECAY = 0.001  # lambda
        VERBOSE_TIME = 10

    n_row, n_column, n_band = img.shape
    img_scaled = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape(img.shape)

    # perform PCA
    pca = PCA(n_components=nb_comps)
    img = pca.fit_transform(img_scaled.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, nb_comps))
    print('pca shape: %s, percentage: %s' % (img.shape, np.sum(pca.explained_variance_ratio_)))
    x_patches, y_ = p.get_HSI_patches(img, gt, (NEIGHBORING_SIZE, NEIGHBORING_SIZE))  # x_patch=(n_samples, n_width, n_height, n_band)
    print(np.unique(y_))
    # perform ZCA whitening
    # x_patches = minmax_scale(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
    # x_patches, _, _ = p_Cora.zca_whitening(x_patches, epsilon=10.)
    x_patches = minmax_scale(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
    print('img shape:', img.shape)
    print('img_patches_nonzero:', x_patches.shape)
    n_samples, n_width, n_height, n_band = x_patches.shape

    y = p.standardize_label(y_)
    # x_patches, y = order_sam_for_diag(x_patches, y)

    # import scipy.io as scio
    # scio.savemat('Houston.mat', {'img': x_patches, 'gt': y})

    print('x_patches shape: %s, labels: %s' % (x_patches.shape, np.unique(y)))

    N_CLASSES = np.unique(y).shape[0]  # wuhan : 5  Pavia : 6  Indian : 8  KSC : 10  SalinasA : 6 PaviaU : 8
    time_start = time.clock()
    model = GRegConvAE(EPOCH, N_CLASSES, im_, lr=LEARNING_RATE, reg_lap=REG_LAP, reg_latent=REG_LATENT,
                       weight_decay=WEIGHT_DECAY, verb_per_iter=VERBOSE_TIME, random_state=SEED)
    model.fit(x_patches, y)
    # y_pre, acc = model.predict_from_model(x_patches_5d, y)
    # loss_, y_pre = model.predict(x_patches_5d)
    # acc = model.cluster_accuracy(y, y_pre)
    # print('predicted acc = ', acc)
    run_time = round(time.clock() - time_start, 3)
    print('running time', run_time)


# tensorboard --logdir=F:\Python\DeepLearning\GraphRegularizedConvAE\logs

# import numpy as np
# import matplotlib.pyplot as plt
# npz = np.load('F:\Python\DeepLearning\GraphRegularizedConvAE\history.npz')
# loss = npz['loss']
# plt.plot(loss)