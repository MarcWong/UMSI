import os
import numpy as np
import json

def load_datasets_singleduration(dataset, bp='/path/to/your/datasets', return_test=False):
    fix_as_mat=False
    fix_key=None

    if dataset == 'gdi':
        print('Using GDI')

        uses_fix =False
        has_classes = False

        img_path_train = os.path.join(bp,'GDI/gd_train')
        imp_path_train = os.path.join(bp,'GDI/gd_imp_train')
        img_path_val = os.path.join(bp,'GDI/gd_val')
        imp_path_val = os.path.join(bp,'GDI/gd_imp_val')

        img_filenames_train = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        imp_filenames_train = sorted([os.path.join(imp_path_train, f) for f in os.listdir(imp_path_train)])
        img_filenames_val = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        imp_filenames_val = sorted([os.path.join(imp_path_val, f) for f in os.listdir(imp_path_val)])

        # Dummy variables
        fix_filenames_train = None #np.array([])
        fix_filenames_val = None #np.array([])

    elif dataset == 'UMSI_SALICON':
        print('Using SALICON (no fixation coords)')

        uses_fix=False
        has_classes = True

        img_path_train = os.path.join(bp, 'Salicon', 'train')
        imp_path_train = os.path.join(bp, 'Salicon', 'train_maps')

        img_path_val = os.path.join(bp, 'Salicon', 'val')
        imp_path_val = os.path.join(bp, 'Salicon', 'val_maps')

        img_path_test = os.path.join(bp, 'Salicon', 'test')

        img_filenames_train = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        imp_filenames_train = sorted([os.path.join(imp_path_train, f) for f in os.listdir(imp_path_train)])

        img_filenames_val = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        imp_filenames_val = sorted([os.path.join(imp_path_val, f) for f in os.listdir(imp_path_val)])

        img_filenames_test = sorted([os.path.join(img_path_test, f) for f in os.listdir(img_path_test)])

        fix_filenames_train = None #np.array(['dummy']*len(img_filenames_train))
        fix_filenames_val = None #np.array(['dummy']*len(img_filenames_val))


        print('Length of loaded files:')
        print('train images:', len(img_filenames_train))
        print('train maps:', len(imp_filenames_train))
        print('val images:', len(img_filenames_val))
        print('val maps:', len(imp_filenames_val))
        print('test images', len(img_filenames_test))

    elif dataset == 'imp1k':
        print('Using imp1k')

        k = 40
        uses_fix=False
        has_classes=True

        img_path = os.path.join(bp, 'imp1k', 'imgs')
        imp_path = os.path.join(bp, 'imp1k', 'maps')

        img_filenames_train = np.array([])
        imp_filenames_train = np.array([])
        img_filenames_val = np.array([])
        imp_filenames_val = np.array([])

        use_tts_file = False

        if not use_tts_file:
            for f in os.listdir(img_path):
                print('Categ:',f)
                imgs = sorted([os.path.join(img_path, f, i) for i in os.listdir(os.path.join(img_path,f)) if i.endswith(('.png','.jpg'))])
                maps = sorted([os.path.join(imp_path, f, i) for i in os.listdir(os.path.join(imp_path,f)) if i.endswith(('.png','.jpg'))])

                idxs = list(range(len(imgs)))
                np.random.shuffle(idxs)
                imgs = np.array(imgs)[idxs]
                maps = np.array(maps)[idxs]

                img_filenames_train = np.concatenate([img_filenames_train,imgs[:-k]], axis=None)
                img_filenames_val = np.concatenate([img_filenames_val,imgs[-k:]], axis=None)
                imp_filenames_train = np.concatenate([imp_filenames_train,maps[:-k]], axis=None)
                imp_filenames_val = np.concatenate([imp_filenames_val,maps[-k:]], axis=None)

        else:
            with open(os.path.join(bp, 'imp1k', 'train_test_split_imp1k.json'), 'r') as f:
                tt_s = json.load(f)

            train_names = tt_s[0]
            test_names = tt_s[1]

            img_filenames_train = [os.path.join(img_path, n) for n in train_names]
            imp_filenames_train = [os.path.join(imp_path, n) for n in train_names]

            img_filenames_val = [os.path.join(img_path, n) for n in test_names]
            imp_filenames_val = [os.path.join(imp_path, n) for n in test_names]


        # Dummy variables
        fix_filenames_train = None #np.array(['dummy']*len(img_filenames_train))
        fix_filenames_val = None #np.array(['dummy']*len(img_filenames_val))

    elif dataset == 'SalChartQA':
        print('Using SalChartQA')

        uses_fix =False
        has_classes = False

        img_path_train = os.path.join(bp,'SalChartQA-MD/train/raw_img')
        imp_path_train = os.path.join(bp,'SalChartQA-MD/train/heatmaps_accum/10000')
        img_path_val = os.path.join(bp,'SalChartQA-MD/val/raw_img')
        imp_path_val = os.path.join(bp,'SalChartQA-MD/val/heatmaps_accum/10000')

        img_filenames_train = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        imp_filenames_train = sorted([os.path.join(imp_path_train, f) for f in os.listdir(imp_path_train)])
        img_filenames_val = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        imp_filenames_val = sorted([os.path.join(imp_path_val, f) for f in os.listdir(imp_path_val)])

        # Dummy variables
        fix_filenames_train = None #np.array([])
        fix_filenames_val = None #np.array([])



    print('Length of loaded files:')
    print('train images:', len(img_filenames_train))
    print('train maps:', len(imp_filenames_train))
    print('val images:', len(img_filenames_val))
    print('val maps:', len(imp_filenames_val))
    if return_test:
        print('test images', len(img_filenames_test))
    
    if fix_filenames_train and fix_filenames_val:
        print('train fixs:', len(fix_filenames_train))
        print('val fixs:', len(fix_filenames_val))
#        print('train fixcoords:', len(fixcoords_filenames_train))
#        print('val fixcoords:', len(fixcoords_filenames_val))


    if return_test:
        return img_filenames_train, imp_filenames_train, fix_filenames_train, img_filenames_val, imp_filenames_val, fix_filenames_val, img_filenames_test, uses_fix, fix_as_mat, fix_key, has_classes
    else:
        return img_filenames_train, imp_filenames_train, fix_filenames_train, img_filenames_val, imp_filenames_val, fix_filenames_val, uses_fix, fix_as_mat, fix_key, has_classes
