import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder
from framework.models_pytorch import move_data_to_gpu
import framework.config as config
from sklearn import metrics
from framework.earlystop import EarlyStopping



def forward(model, generate_func, cuda, return_names = False):
    output_scene, output_event = [], []
    output_ISOPls, output_ISOEvs = [], []
    output_pleasant, output_eventful, output_chaotic, output_vibrant = [], [], [], []
    output_uneventful, output_calm, output_annoying, output_monotonous = [], [], [], []

    label_scene, label_event = [], []
    label_ISOPls, label_ISOEvs = [], []
    label_pleasant, label_eventful, label_chaotic, label_vibrant = [], [], [], []
    label_uneventful, label_calm, label_annoying, label_monotonous = [], [], [], []

    audio_names = []
    # Evaluate on mini-batch
    for num, data in enumerate(generate_func):
        # print(num)
        if return_names:
            (batch_x, batch_x_loudness, batch_scene, batch_event, batch_graph, batch_ISOPls, batch_ISOEvs,
             batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
             batch_uneventful, batch_calm, batch_annoying, batch_monotonous, names) = data
            audio_names.append(names)
        else:
            (batch_x, batch_x_loudness, batch_scene, batch_event, batch_graph, batch_ISOPls, batch_ISOEvs,
             batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
             batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_x_loudness = move_data_to_gpu(batch_x_loudness, cuda)

        model.eval()
        with torch.no_grad():
            scene, event, ISOPls, ISOEvs, \
            pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model(batch_x, batch_x_loudness, batch_graph)
            # print(scene.shape, event.shape, ISOPls.shape, ISOEvs.shape,
            #       pleasant.shape, eventful.shape, chaotic.shape, vibrant.shape,
            #       uneventful.shape, calm.shape, annoying.shape, monotonous.shape)

            event = F.sigmoid(event)

            output_scene.append(scene.data.cpu().numpy())
            output_event.append(event.data.cpu().numpy())

            output_ISOPls.append(ISOPls.data.cpu().numpy())
            output_ISOEvs.append(ISOEvs.data.cpu().numpy())

            output_pleasant.append(pleasant.data.cpu().numpy())
            output_eventful.append(eventful.data.cpu().numpy())
            output_chaotic.append(chaotic.data.cpu().numpy())
            output_vibrant.append(vibrant.data.cpu().numpy())
            output_uneventful.append(uneventful.data.cpu().numpy())
            output_calm.append(calm.data.cpu().numpy())
            output_annoying.append(annoying.data.cpu().numpy())
            output_monotonous.append(monotonous.data.cpu().numpy())
            # print('output_monotonous: ', output_monotonous)

            # ------------------------- labels -------------------------------------------------------------------------
            label_scene.append(batch_scene)
            label_event.append(batch_event)

            label_ISOPls.append(batch_ISOPls)
            label_ISOEvs.append(batch_ISOEvs)

            label_pleasant.append(batch_pleasant)
            label_eventful.append(batch_eventful)
            label_chaotic.append(batch_chaotic)
            label_vibrant.append(batch_vibrant)
            label_uneventful.append(batch_uneventful)
            label_calm.append(batch_calm)
            label_annoying.append(batch_annoying)
            label_monotonous.append(batch_monotonous)

    dict = {}

    if return_names:
        dict['audio_names'] = np.concatenate(audio_names, axis=0)

    dict['output_scene'] = np.concatenate(output_scene, axis=0)
    dict['output_event'] = np.concatenate(output_event, axis=0)

    dict['output_ISOPls'] = np.concatenate(output_ISOPls, axis=0)
    dict['output_ISOEvs'] = np.concatenate(output_ISOEvs, axis=0)

    dict['output_pleasant'] = np.concatenate(output_pleasant, axis=0)
    dict['output_eventful'] = np.concatenate(output_eventful, axis=0)
    dict['output_chaotic'] = np.concatenate(output_chaotic, axis=0)
    dict['output_vibrant'] = np.concatenate(output_vibrant, axis=0)
    dict['output_uneventful'] = np.concatenate(output_uneventful, axis=0)
    dict['output_calm'] = np.concatenate(output_calm, axis=0)
    dict['output_annoying'] = np.concatenate(output_annoying, axis=0)
    dict['output_monotonous'] = np.concatenate(output_monotonous, axis=0)

    # print(dict)
    # ----------------------------- labels -------------------------------------------------------------------------
    dict['label_scene'] = np.concatenate(label_scene, axis=0)
    dict['label_event'] = np.concatenate(label_event, axis=0)

    dict['label_ISOPls'] = np.concatenate(label_ISOPls, axis=0)
    dict['label_ISOEvs'] = np.concatenate(label_ISOEvs, axis=0)

    dict['label_pleasant'] = np.concatenate(label_pleasant, axis=0)
    dict['label_eventful'] = np.concatenate(label_eventful, axis=0)
    dict['label_chaotic'] = np.concatenate(label_chaotic, axis=0)
    dict['label_vibrant'] = np.concatenate(label_vibrant, axis=0)
    dict['label_uneventful'] = np.concatenate(label_uneventful, axis=0)
    dict['label_calm'] = np.concatenate(label_calm, axis=0)
    dict['label_annoying'] = np.concatenate(label_annoying, axis=0)
    dict['label_monotonous'] = np.concatenate(label_monotonous, axis=0)

    return dict


def cal_auc(targets_event, outputs_event):
    # print(targets_event)
    # print(outputs_event)
    #
    # print(targets_event.shape)
    # print(outputs_event.shape)
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch


def cal_softmax_classification_accuracy(target, predict, average=None, eps=1e-8):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """
    # print(target)
    # print(predict)
    classes_num = predict.shape[-1]

    predict = np.argmax(predict, axis=-1)  # (audios_num,)
    samples_num = len(target)


    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / (total + eps)

    if average == 'each_class':
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')



def evaluate(model, generate_func, cuda):
    # Forward
    dict = forward(model=model, generate_func=generate_func, cuda=cuda)

    # mse loss
    ISOPls_mse = metrics.mean_squared_error(dict['label_ISOPls'], dict['output_ISOPls'])
    ISOEvs_mse = metrics.mean_squared_error(dict['label_ISOEvs'], dict['output_ISOEvs'])
    # rate_rmse_loss = metrics.mean_squared_error(targets, predictions, squared=False)
    # squared: If True returns MSE value, if False returns RMSE value.
    # rmse

    # AUC
    event_auc = cal_auc(dict['label_event'], dict['output_event'])

    # softmax classification acc
    scene_acc = cal_softmax_classification_accuracy(dict['label_scene'], dict['output_scene'], average = 'macro')

    pleasant_mse = metrics.mean_squared_error(dict['label_pleasant'], dict['output_pleasant'])
    eventful_mse = metrics.mean_squared_error(dict['label_eventful'], dict['output_eventful'])
    chaotic_mse = metrics.mean_squared_error(dict['label_chaotic'], dict['output_chaotic'])
    vibrant_mse = metrics.mean_squared_error(dict['label_vibrant'], dict['output_vibrant'])
    uneventful_mse = metrics.mean_squared_error(dict['label_uneventful'], dict['output_uneventful'])
    calm_mse = metrics.mean_squared_error(dict['label_calm'], dict['output_calm'])
    annoying_mse = metrics.mean_squared_error(dict['label_annoying'], dict['output_annoying'])
    monotonous_mse = metrics.mean_squared_error(dict['label_monotonous'], dict['output_monotonous'])

    return scene_acc, event_auc, ISOPls_mse, ISOEvs_mse, \
           pleasant_mse, eventful_mse, chaotic_mse, vibrant_mse, uneventful_mse, calm_mse, annoying_mse, monotonous_mse



def Training_early_stopping(generator, model, models_dir, batch_size, monitor, cuda=config.cuda,
                            epochs=config.epochs, patience=10, lr_init=config.lr_init, alpha=None):
    create_folder(models_dir)

    optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

    # ------------------------------------------------------------------------------------------------------------------

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    sample_num = len(generator.train_scene_labels)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = one_epoch
    print('validating every: ', check_iter, ' iteration')

    # initialize the early_stopping object
    model_path = os.path.join(models_dir, 'early_stopping_' + monitor + config.endswith)
    early_stopping_mse_loss = EarlyStopping(model_path, decrease=True, patience=patience, verbose=True)

    training_start_time = time.time()
    for iteration, all_data in enumerate(generator.generate_train()):

        (batch_x, batch_x_loudness, batch_scene, batch_sound_masker, batch_graph, batch_ISOPls, batch_ISOEvs,
         batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
         batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = all_data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_x_loudness = move_data_to_gpu(batch_x_loudness, cuda)

        batch_scene = move_data_to_gpu(batch_scene, cuda)
        batch_sound_masker = move_data_to_gpu(batch_sound_masker, cuda)

        # MSE
        batch_ISOPls = move_data_to_gpu(batch_ISOPls, cuda)
        batch_ISOEvs = move_data_to_gpu(batch_ISOEvs, cuda)

        # MSE
        batch_pleasant = move_data_to_gpu(batch_pleasant, cuda, using_float=True)
        batch_eventful = move_data_to_gpu(batch_eventful, cuda, using_float=True)
        batch_chaotic = move_data_to_gpu(batch_chaotic, cuda, using_float=True)
        batch_vibrant = move_data_to_gpu(batch_vibrant, cuda, using_float=True)
        batch_uneventful = move_data_to_gpu(batch_uneventful, cuda, using_float=True)
        batch_calm = move_data_to_gpu(batch_calm, cuda, using_float=True)
        batch_annoying = move_data_to_gpu(batch_annoying, cuda, using_float=True)
        batch_monotonous = move_data_to_gpu(batch_monotonous, cuda, using_float=True)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        scene, event, ISOPls, ISOEvs, \
        pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model(batch_x, batch_x_loudness, batch_graph)

        loss_scene = F.nll_loss(F.log_softmax(scene, dim=-1), batch_scene)
        loss_event = bce_loss(F.sigmoid(event), batch_sound_masker)

        loss_ISOPls = mse_loss(ISOPls, batch_ISOPls)
        loss_ISOEvs = mse_loss(ISOEvs, batch_ISOEvs)

        loss_pleasant = mse_loss(pleasant, batch_pleasant)
        loss_eventful = mse_loss(eventful, batch_eventful)
        loss_chaotic = mse_loss(chaotic, batch_chaotic)
        loss_vibrant = mse_loss(vibrant, batch_vibrant)
        loss_uneventful = mse_loss(uneventful, batch_uneventful)
        loss_calm = mse_loss(calm, batch_calm)
        loss_annoying = mse_loss(annoying, batch_annoying)
        loss_monotonous = mse_loss(monotonous, batch_monotonous)

        if alpha is not None:
            if type(alpha[0]) == str:
                alpha = [float(each) for each in alpha]
                loss_common = alpha[0] * loss_scene + alpha[1] * loss_event + \
                              alpha[2] * loss_ISOPls + alpha[3] * loss_ISOEvs \
                              + alpha[4] * loss_pleasant + alpha[5] * loss_eventful + alpha[6] * loss_chaotic + alpha[7] * loss_vibrant \
                              + alpha[8] * loss_uneventful + alpha[9] * loss_calm + alpha[10] * loss_annoying + alpha[11] * loss_monotonous
            else:
                loss_common = alpha[0] * loss_scene + alpha[1] * loss_event + alpha[2] * loss_ISOPls + alpha[3] * loss_ISOEvs \
                              + alpha[4] * loss_pleasant + alpha[5] * loss_eventful + alpha[6] * loss_chaotic + alpha[7] * loss_vibrant \
                              + alpha[8] * loss_uneventful + alpha[9] * loss_calm + alpha[10] * loss_annoying + alpha[11] * loss_monotonous

        ############################################################################################
        loss_common.backward()  # calculate the gradient can apply gradient modification
        optimizer.step()  # apply gradient step

        Epoch = iteration / one_epoch
        print('epoch: ', '%.3f' % (Epoch), 'loss: %.3f' % float(loss_common),
              'scene: %.3f' % float(loss_scene), 'event: %.3f' % float(loss_event),
              'ISOPls: %.3f' % float(loss_ISOPls), 'ISOEvs: %.3f' % float(loss_ISOEvs),

              'plea: %.3f' % float(loss_pleasant), 'eventf: %.3f' % float(loss_eventful),
              'chao: %.3f' % float(loss_chaotic), 'vib: %.3f' % float(loss_vibrant),
              'uneve: %.3f' % float(loss_uneventful), 'calm: %.3f' % float(loss_calm),
              'ann: %.3f' % float(loss_annoying), 'mono: %.3f' % float(loss_monotonous))

        if iteration % check_iter == 0 and iteration > 1:
            train_fin_time = time.time()
            # Generate function
            generate_func = generator.generate_validate(data_type='validate')
            val_scene_acc, val_event_auc, val_ISOPls_mse, val_ISOEvs_mse, \
            val_pleasant_mse, val_eventful_mse, val_chaotic_mse, val_vibrant_mse, \
            val_uneventful_mse, val_calm_mse, val_annoying_mse, val_monotonous_mse = evaluate(model=model,
                                                              generate_func=generate_func,
                                                              cuda=cuda)

            print('E: ', '%.3f' % (Epoch),
                  'val_scene: %.3f' % float(val_scene_acc), 'val_event: %.3f' % float(val_event_auc),
                  'val_ISOP: %.3f' % float(val_ISOPls_mse), 'val_ISOE: %.3f' % float(val_ISOEvs_mse),

                  'val_plea: %.3f' % float(val_pleasant_mse), 'val_even: %.3f' % float(val_eventful_mse),
                  'val_chao: %.3f' % float(val_chaotic_mse), 'val_vibr: %.3f' % float(val_vibrant_mse),
                  'val_uneve: %.3f' % float(val_uneventful_mse), 'val_calm: %.3f' % float(val_calm_mse),
                  'val_anno: %.3f' % float(val_annoying_mse), 'val_mono: %.3f' % float(val_monotonous_mse))

            train_time = train_fin_time - train_bgn_time

            validation_end_time = time.time()
            validate_time = validation_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))
            #------------------------ validation done ------------------------------------------------------------------

            # -------- early stop---------------------------------------------------------------------------------------
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if Epoch > 10:
                # if monitor == 'event':
                #     early_stopping_auc(val_event_auc, model)
                # if monitor == 'scene':
                #     early_stopping_auc(val_scene_acc, model)
                if monitor == 'ISOPls':
                    early_stopping_mse_loss(val_ISOPls_mse, model)
                if monitor == 'ISOEvs':
                    early_stopping_mse_loss(val_ISOEvs_mse, model)

                if early_stopping_mse_loss.early_stop:
                    finish_time = time.time() - training_start_time
                    print('Model training finish time: {:.3f} s,'.format(finish_time))
                    print("Early stopping")

                    save_out_dict = {'state_dict': model.state_dict()}
                    save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
                    torch.save(save_out_dict, save_out_path)
                    print('Final model saved to {}'.format(save_out_path))

                    print('Model training finish time: {:.3f} s,'.format(finish_time))
                    print('Model training finish time: {:.3f} s,'.format(finish_time))
                    print('Model training finish time: {:.3f} s,'.format(finish_time))

                    print('Training is done!!!')

                    break

        # Stop learning
        if iteration > (epochs * one_epoch):
            finish_time = time.time() - training_start_time
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print("All epochs are done.")

            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))

            print('Training is done!!!')

            break






