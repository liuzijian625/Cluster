import numpy as np

def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
        print('count:',count)
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs


def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    labels = np.zeros(inst_num)
    for i, label in idx2lb.items():
        labels[i] = label
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return labels

def read(feat_path, label_path):
    label = read_meta(label_path)
    inst_num = len(label)
    feat = read_probs(feat_path, inst_num=inst_num, feat_dim=256)
    print("feature shape:{}".format(feat.shape))
    print("label number:{}".format(len(np.unique(label))))
    return feat, label

if __name__ == "__main__":
    # 读取特征和对应标签方法
    feat_path = './feat.bin'
    label_path = './label.meta'
    feat, label = read(feat_path, label_path)