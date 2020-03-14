import os
from collections import defaultdict
import shutil, sys 

def copytree(src, dst, symlinks = False, ignore = None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def generate_dir_file_map(path):
    dir_files = defaultdict(list)
    with open(path, 'r') as txt:
        files = [l.strip() for l in txt.readlines()]
        for f in files:
            dir_name, id = f.split('/')
            dir_files[dir_name].append(id + '.jpg')
    return dir_files

def ignore_train(d, filenames):
    print(d)
    subdir = d.split('/')[-1]
    to_ignore = train_dir_files[subdir]
    return to_ignore

def ignore_test(d, filenames):
    print(d)
    subdir = d.split('/')[-1]
    to_ignore = test_dir_files[subdir]
    return to_ignore

if len(sys.argv) > 1:
    root_path = sys.argv[1] +'/'
else:
    root_path =''

if not os.path.isdir(root_path+'food-101/test') or not os.path.isdir(root_path+'food-101/train'):
    train_dir_files = generate_dir_file_map('food-101/meta/train.txt')
    test_dir_files = generate_dir_file_map('food-101/meta/test.txt')
    copytree('food-101/images',root_path+'food-101/test', ignore=ignore_train)
    copytree('food-101/images', root_path+'food-101/train', ignore=ignore_test)
else:
    print('Train/Test files already copied into separate folders.')