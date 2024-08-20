from typing import List

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os, sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr

from classifier_sprites import classifier_Sprite_all
from dataloader import data_sprites
import flags
from networks import Encoder, Decoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def define_args():
    parser = argparse.ArgumentParser(description="Sprites SKD")

    # general
    # parser.add_argument('--cuda', action='store_false')
    # parser.add_argument('--epochs', type=int, default=1000)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")
    parser.add_argument('--num_dim', type=int, default=10, help="total dimension of latent space")  # L

    # data
    # parser.add_argument("--dataset_path", default='/cs/cs_groups/azencot_group/datasets/SPRITES_ICML/datasetICML/')
    # parser.add_argument("--dataset", default='Sprites')
    # parser.add_argument('--batch_size', type=int, default=32, metavar='N')

    # model
    # parser.add_argument('--arch', type=str, default='KoopmanCNN')
    # parser.add_argument('--conv_dim', type=int, default=32)
    # parser.add_argument('--dropout', type=float, default=0.2)
    # parser.add_argument('--rnn', type=str, default='both',
    #                     help='encoder decoder LSTM strengths. Can be: "none", "encoder","decoder", "both"')
    # parser.add_argument('--k_dim', type=int, default=40)
    # parser.add_argument('--hidden_dim', type=int, default=40,
    #                     help='the hidden dimension of the output decoder lstm')
    # parser.add_argument('--lstm_dec_bi', type=bool, default=False)  # nimrod added

    # loss params
    # parser.add_argument('--w_rec', type=float, default=15.0)
    # parser.add_argument('--w_pred', type=float, default=1.0)
    # parser.add_argument('--w_eigs', type=float, default=1.0)

    # eigen values system params
    # parser.add_argument('--static_size', type=int, default=8)
    # parser.add_argument('--static_mode', type=str, default='norm', choices=['norm', 'real', 'ball'])
    # parser.add_argument('--dynamic_mode', type=str, default='ball',
    #                     choices=['strict', 'thresh', 'ball', 'real', 'none'])

    # thresholds
    # parser.add_argument('--ball_thresh', type=float, default=0.6)  # related to 'ball' dynamic mode
    # parser.add_argument('--dynamic_thresh', type=float, default=0.5)  # related to 'thresh', 'real'
    # parser.add_argument('--eigs_thresh', type=float, default=.5)  # related to 'norm' static mode loss

    # other
    # parser.add_argument('--noise', type=str, default='none', help='adding blur to the sample (in the pixel space')

    # parameters for the classifier
    parser.add_argument('--g_dim', default=128, type=int,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
    parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
    parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')

    # args for testing
    parser.add_argument('--exploration_type', type=str, default='supervised', choices=['supervised', 'unsupervised'])
    # parser.add_argument('--classifier_type', type=str, default='decision_tree', choices=['linear', 'decision_tree'])

    return parser


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def explore_latent_space_and_return_mapping(latent_codes: List[np.ndarray], attributes: List[List[int]],
                                            classifier_type: str = 'linear') -> List[List[int]]:
    """
    This function takes a list of latent codes, and for each label in the attribute list, it trains a classifier
    to predict the label from the latent code. For each latent code coordinate, it collects its relative importance
    score for each label. Then it assigns each coordinate to the label with the highest importance score.

    Args:
    - latent_codes: A list of tensors, each tensor is a latent code.
    - attributes: A list of lists, where each list contains the attributes of the corresponding latent code.
    - classifier_type: The type of classifier to use, either 'linear' (Logistic Regression) or 'decision_tree'.

    Returns:
    - mapping: A list of lists where each sublist corresponds to the most important label for each latent code coordinate.
    """

    # Convert latent codes to a numpy array
    latent_codes_np = np.array(latent_codes)
    # Convert attributes to numpy array
    attributes_np = np.array(attributes)

    num_latent_dims = latent_codes_np.shape[1]

    num_labels = attributes_np.shape[1]

    importance_scores = np.zeros((num_latent_dims, num_labels))

    for label_idx in range(num_labels):
        # Select the classifier
        if classifier_type == 'linear':
            classifier = LogisticRegression()
        elif classifier_type == 'decision_tree':
            classifier = DecisionTreeClassifier(max_depth=8)
        else:
            raise ValueError("Unsupported classifier_type. Choose either 'linear' or 'decision_tree'.")

        # Train the classifier on the latent codes to predict the current label
        classifier.fit(latent_codes_np, attributes_np[:, label_idx])
        # print the accuracy
        print(f"Accuracy for label {label_idx}: {classifier.score(latent_codes_np, attributes_np[:, label_idx])}")

        # Collect the importance scores
        if classifier_type == 'linear':
            # For linear classifier, importance is the absolute value of the coefficients
            normalized_coefs = (np.abs(classifier.coef_) / np.sum(np.abs(classifier.coef_))).sum(axis=0)
            importance_scores[:, label_idx] = normalized_coefs
        elif classifier_type == 'decision_tree':
            normalized_coefs = classifier.feature_importances_ / np.sum(classifier.feature_importances_)
            # For decision tree, importance is the feature_importances_ attribute
            importance_scores[:, label_idx] = normalized_coefs

    # Assign each coordinate to the label with the highest importance score
    mapping = np.argmax(importance_scores, axis=1)

    # if all importance scores are zero, assign the coordinate -1
    mapping[np.all(importance_scores == 0, axis=1)] = -1

    # Convert the mapping to a list of lists
    return mapping.tolist()


def unsupervised_latent_mapping(latent_codes: List[np.ndarray], group_num=5) -> List[List[int]]:
    """
    Given a list of latent codes, the function calculates the correlation between each coordinate in the latent codes,
    and then groups the coordinates into `group_num` groups based on the correlation between them.

    Args:
    - latent_codes: A list of latent codes where each latent code is a numpy array.
    - group_num: The number of groups to cluster the coordinates into.

    Returns:
    - A list of lists where each sublist contains the indices of the coordinates belonging to the same group.
    """

    # Convert latent codes to a numpy array
    latent_codes_np = np.array(latent_codes)

    # Calculate the correlation matrix between the latent code coordinates
    num_coordinates = latent_codes_np.shape[1]
    correlation_matrix = np.zeros((num_coordinates, num_coordinates))

    for i in range(num_coordinates):
        for j in range(num_coordinates):
            if i == j:
                correlation_matrix[i, j] = 1.0  # Correlation with itself is 1
            else:
                correlation_matrix[i, j] = pearsonr(latent_codes_np[:, i], latent_codes_np[:, j])[0]

    # Convert the correlation matrix to a distance matrix (1 - correlation)
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Perform hierarchical clustering based on the distance matrix
    clustering = AgglomerativeClustering(n_clusters=group_num, affinity='precomputed', linkage='average')
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Group the coordinates based on the clustering labels
    groups = [[] for _ in range(group_num)]
    for coord_idx, group_label in enumerate(cluster_labels):
        groups[group_label].append(coord_idx)

    return groups


def load_model():
    root = r'/cs_storage/baramit/research/MGP-VAE'
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dict(torch.load(os.path.join(root, 'outputs', 'checkpoints', 'sprites', 'encoder_sprites_debug')))
    decoder.load_state_dict(torch.load(os.path.join(root, 'outputs', 'checkpoints', 'sprites', 'decoder_sprites_debug')))
    encoder.eval()
    decoder.eval()

    return encoder, decoder


def load_data_for_explore_and_test():
    # global test_data
    # global idx, test_data
    # train_data, test_data = load_dataset(args)
    # # reduce the size of test data into k samples
    # k = 1024
    # idx = np.random.choice(2664, k, replace=False)
    # test_data.data = test_data.data[idx]
    # test_data.A_label = test_data.A_label[idx]
    # test_data.D_label = test_data.D_label[idx]
    # test_data.N = k
    test_data = data_sprites(r'/cs/cs_groups/azencot_group/datasets/SMD/sprites/sprites_test.npz', return_labels=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=16,  # 128
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    # x, label_A, label_D = reorder(torch.tensor(test_data.data)), test_data.A_label[:, 0], test_data.D_label[:, 0]
    # x = x.to(args.device)

    # transfer label_a and label_d from one hot encoding to a single label
    # label_A = np.argmax(label_A, axis=-1)
    # label_D = np.argmax(label_D, axis=1)
    # append label_d to label_a
    # labels = np.column_stack((label_A, label_D))

    return test_data, test_loader


def extract_latent_code(encoder, data_loader, device):
    # global D, ZL
    # outputs = model(x)
    # _, Ct_te, Z = outputs[0], outputs[-1], outputs[2]
    # Z = t_to_np(Z.reshape(x.shape[0], 8, 40))  # 8 its the time steps in this dataset and 40 is the latent dimension
    # C = t_to_np(Ct_te)
    # eig
    # D, V = np.linalg.eig(C)
    # U = np.linalg.inv(V)
    # project onto V
    # ZL = np.real((Z @ V).mean(axis=1))  # create a single latent code scaler for each sample
    Xs = []
    Ys = []
    for X, Y in data_loader:
        X = X.to(device)
        X1, KL1, muL1, det_q1 = encoder(X)
        Xs.append(X1.detach().cpu().numpy())
        Ys.append(Y.detach().cpu().numpy())
    Xs = np.concatenate(Xs, axis=0)
    Xs = np.mean(Xs, axis=1)
    Ys = np.concatenate(Ys, axis=0)
    return Xs, Ys




def check_cls_specific_indexes(encoder, decoder, classifier, test_loader, run, run_type, target_indexes, fix=False, swap=False, device='cuda:0'):
    for epoch in range(1):
        # print("Epoch", epoch)
        encoder.eval()
        decoder.eval()
        mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        _target_indexes = [x for x in range(5) if x not in target_indexes]
        for X, Y in test_loader:
            X = X.to(device)
            idxs = np.arange(flags.BATCH_SIZE)
            np.random.shuffle(idxs)
            _X = X[idxs]
            _Y = Y[idxs]
            if fix:
                if swap:
                    Z, KL, mu, det_q = encoder(X)
                    recon_x = decoder(Z)[idxs]
                    Z1, Z2 = Z[idxs[:flags.BATCH_SIZE//2]], Z[idxs[flags.BATCH_SIZE//2:]]
                    # swap all but target_indexes:
                    _Z1 = Z1[:, _target_indexes].clone()
                    Z1[:, _target_indexes] = Z2[:, _target_indexes]
                    Z2[:, _target_indexes] = _Z1
                    _Z = torch.cat([Z1, Z2], dim=0)
                    recon_x_sample = decoder(_Z)
                else:
                    raise NotImplementedError
            # else:
            #     if swap:
            #         recon_x_sample, recon_x = cdsvae.forward_swap_for_classification2_specific_I(target_indexes, x,
            #                                                                                        fix=False)
            #     else:
            #         recon_x_sample, recon_x = cdsvae.forward_sample_for_classification2_specific_I(target_indexes, x,
            #                                                                                        fix=False)

            with torch.no_grad():
                pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = classifier(_X)
                pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = classifier(recon_x_sample)
                pred_action3, pred_skin3, pred_pant3, pred_top3, pred_hair3 = classifier(recon_x)

                pred1 = F.softmax(pred_action1, dim=1)
                pred2 = F.softmax(pred_action2, dim=1)
                pred3 = F.softmax(pred_action3, dim=1)

            label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
            label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)
            label2_all.append(label2)

            pred1_all.append(pred1.detach().cpu().numpy())
            # pred2_all.append(pred2.detach().cpu().numpy())
            _Y = _Y.detach().cpu().numpy()
            label_gt.append(np.argmax(_Y, axis=1))

            def count_D(pred, label, mode=1):
                return (pred // mode) == (label // mode)
            # action
            acc0_sample = (np.argmax(pred_action2.detach().cpu().numpy(), axis=1) == _Y[:, 0]).mean()
            # skin
            acc1_sample = (np.argmax(pred_skin2.detach().cpu().numpy(), axis=1) == _Y[:, 1]).mean()
            # pant
            acc2_sample = (np.argmax(pred_pant2.detach().cpu().numpy(), axis=1) == _Y[:, 2]).mean()
            # top
            acc3_sample = (np.argmax(pred_top2.detach().cpu().numpy(), axis=1) == _Y[:, 3]).mean()
            # hair
            acc4_sample = (np.argmax(pred_hair2.detach().cpu().numpy(), axis=1) == _Y[:, 4]).mean()
            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample
            mean_acc2_sample += acc2_sample
            mean_acc3_sample += acc3_sample
            mean_acc4_sample += acc4_sample

        print(
            'action_Acc: {:.2f}% skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
                mean_acc0_sample / len(test_loader) * 100,
                mean_acc1_sample / len(test_loader) * 100, mean_acc2_sample / len(test_loader) * 100,
                mean_acc3_sample / len(test_loader) * 100, mean_acc4_sample / len(test_loader) * 100))

        label2_all = np.hstack(label2_all)
        label_gt = np.hstack(label_gt)
        # pred1_all = np.vstack(pred1_all)
        # pred2_all = np.vstack(pred2_all)

        # acc = (label_gt == label2_all).mean()
        # kl = KL_divergence(pred2_all, pred1_all)

        # nSample_per_cls = min([(label_gt == i).sum() for i in np.unique(label_gt)])
        # index = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
        # pred2_selected = pred2_all[index]

        # IS = inception_score(pred2_selected)
        # H_yx = entropy_Hyx(pred2_selected)
        # H_y = entropy_Hy(pred2_selected)

        # print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc * 100, 0, 0, 0, 0))
    action_Acc = mean_acc0_sample / len(test_loader) * 100
    skin_Acc = mean_acc1_sample / len(test_loader) * 100
    pant_Acc = mean_acc2_sample / len(test_loader) * 100
    top_Acc = mean_acc3_sample / len(test_loader) * 100
    hair_Acc = mean_acc4_sample / len(test_loader) * 100

    return action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc

def full_experiments(seed):
    # todo - repeat the main experiment 5 times and calculate the mean and variance of each table and metric
    pass

if __name__ == "__main__":
    # TODO - make this experiment more 5 times with different seeds !!! and calculate the mean and variance
    # --- load configurations ---
    parser = define_args()
    args = parser.parse_args()
    args.device = set_seed_device(args.seed)
    device = args.device
    # --- load here you model ---
    encoder, decoder = load_model()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # --- load your data (large batch) and labels here ---
    test_data, test_loader = load_data_for_explore_and_test()

    #  --- extract latent codes from you model---
    ZL, Y = extract_latent_code(encoder, test_loader, device)

    #  --- Latent exploration !!! ---
    # todo - maybe add more options or classifiers
    if args.exploration_type == 'supervised':
        mappings = explore_latent_space_and_return_mapping(latent_codes=ZL,
                                                           attributes=Y,
                                                           classifier_type='linear')
    else:
        # todo - if using unsupervised - need to align the table with the labels
        mappings = unsupervised_latent_mapping(latent_codes=list(ZL), group_num=5)

    if args.exploration_type == 'supervised':
        map_idx_to_label = {}
        for idx, label in enumerate(mappings):
            if label != -1:
                map_idx_to_label[idx] = label

        map_label_to_idx = {}
        for idx, label in map_idx_to_label.items():
            if label not in map_label_to_idx:
                map_label_to_idx[label] = []
            map_label_to_idx[label].append(idx)

    else:
        map_label_to_idx = {}
        for label, group in enumerate(mappings):
            map_label_to_idx[label] = group

    # --- load classifier for testing of sprites --- #
    classifier = classifier_Sprite_all(args)
    args.resume = 'sprite_judge.tar'
    loaded_dict = torch.load(args.resume)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()
    print('loader classifier from: ', args.resume)

    print('--- start multifactor classification --- \n')
    label_to_name_dict = {
        0: 'skin',
        1: 'pant',
        2: 'top',
        3: 'hair',
        4: 'action'
    }
    df = pd.DataFrame(columns=['action', 'skin', 'pant', 'top', 'hair'],
                      index=['action', 'skin', 'pant', 'top', 'hair'])

    #  --- Swap generation evaluation !!! ---
    for label in map_label_to_idx:
        print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
        action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc = check_cls_specific_indexes(encoder, decoder, classifier, test_loader,
                                                                                       None, 'action',
                                                                                       map_label_to_idx[label],
                                                                                       fix=True,
                                                                                       swap=True)
        df.loc[label_to_name_dict[label]] = [action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc]
        # print df
    # print full table
    print("swap")
    print(df)

    # calculate final score
    scores = df.values
    scores_mask = np.zeros_like(scores)
    random_floor_dict = {'skin': 16.66, 'pant': 16.66, 'top': 16.66, 'hair': 16.66, 'action': 11.11}
    for k in random_floor_dict.keys():
        scores_mask[:, df.columns.get_loc(k)] = random_floor_dict[k]
    # add on scores mask diagonal 100
    np.fill_diagonal(scores_mask, 100)

    off_diagonal = np.where(~np.eye(scores.shape[0], dtype=bool))
    off_diag_score = np.abs(scores[off_diagonal] - scores_mask[off_diagonal]).mean()
    diag_score = np.mean(100 - np.diag(scores))
    final_score = (off_diag_score + diag_score) / 2
    print(f"Final score: {final_score}")

    #  --- Swap generation evaluation !!! ---
    # for label in map_label_to_idx:
    #     print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
    #     action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc = check_cls_specific_indexes(encoder, decoder, classifier, test_loader,
    #                                                                                    None, 'action',
    #                                                                                    map_label_to_idx[label],
    #                                                                                    fix=True, swap=False)
    #     df.loc[label_to_name_dict[label]] = [action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc]
    #     # print df
    # # print full table
    # print("generation swap")
    # print(df)
    #
    # # calculate final score
    # scores = df.values
    # scores_mask = np.zeros_like(scores)
    # random_floor_dict = {'skin': 16.66, 'pant': 16.66, 'top': 16.66, 'hair': 16.66, 'action': 11.11}
    # for k in random_floor_dict.keys():
    #     scores_mask[:, df.columns.get_loc(k)] = random_floor_dict[k]
    # # add on scores mask diagonal 100
    # np.fill_diagonal(scores_mask, 100)
    #
    # off_diagonal = np.where(~np.eye(scores.shape[0], dtype=bool))
    # off_diag_score = np.abs(scores[off_diagonal] - scores_mask[off_diagonal]).mean()
    # diag_score = np.mean(100 - np.diag(scores))
    # final_score = (off_diag_score + diag_score) / 2
    # print(f"Final score: {final_score}")

