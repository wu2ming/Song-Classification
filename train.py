import time
import numpy as np
import torch
import json
from torch import optim
from util import EarlyStopping
from Paras import Para


def train(model, epoch, train_loader, optimizer, loss_function, versatile=True):
    start_time = time.time()
    model = model.train()
    train_loss = 0.
    accuracy = 0.
    batch_num = len(train_loader)
    _index = 0
    for _index, data in enumerate(train_loader):
        spec_input, target = data['mel'], data['tag']

        if Para.cuda:
            spec_input = spec_input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        #forward
        outputs = model(spec_input)

        _, predicted = torch.max(outputs, 1)
        accuracy_value = ((predicted == target.max(1)[1]).sum().item())/outputs.shape[0]
        accuracy += accuracy_value

        #backward
        loss_value = loss_function(outputs,target)
        loss_value.backward()
        #update
        optimizer.step()


        train_loss += loss_value.data.item()

        if versatile:
            if (_index + 1) % Para.log_step == 0:
                elapsed = time.time() - start_time
                print('Epoch{:3d} | {:3d}/{:3d} batches | {:5.2f}ms/ batch | BCE: {:5.4f} | Accuracy: {:5.2f}% |'
                      .format(epoch, _index + 1, batch_num,
                              elapsed * 1000 / (_index + 1),
                              train_loss / (_index + 1),
                              accuracy * 100 / (_index + 1)))

    train_loss /= (_index + 1)
    accuracy /= (_index + 1)

    print('-' * 99)
    print('End of training epoch {:3d} | time: {:5.2f}s | BCE: {:5.4f} | Accuracy: {:5.2f}% |'
          .format(epoch, (time.time() - start_time),
                  train_loss, accuracy * 100))

    return train_loss, accuracy


def validate_test(model, epoch, use_loader, loss_function):
    start_time = time.time()
    model = model.eval()
    v_loss = 0.
    accuracy = 0.
    data_loader_use = use_loader
    _index = 0
    for _index, data in enumerate(data_loader_use):
        spec_input, target = data['mel'], data['tag']

        if Para.cuda:
            spec_input = spec_input.cuda()
            target = target.cuda()

        with torch.no_grad():

            outputs = model(spec_input)
            loss_value = loss_function(outputs,target)
            #compute loss and accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy_value = ((predicted == target.max(1)[1]).sum().item())/outputs.shape[0]
            accuracy += accuracy_value
            v_loss += loss_value.data.item()

    v_loss /= (_index + 1)
    accuracy /= (_index + 1)

    print('End of validation epoch {:3d} | time: {:5.2f}s | BCE: {:5.4f} | Accuracy: {:5.2f}% |'
          .format(epoch, (time.time() - start_time),
                  v_loss, accuracy * 100))
    print('-' * 99)

    return v_loss, accuracy

#main training loop
def main_train(model, train_loader, valid_loader, log_name, save_name, loss_function, lr=Para.learning_rate, epoch_num=Para.epoch_num):
    callback = EarlyStopping(patience = 4, deviation = 0.05, learning= 0.01)
    Para.dataset_len = len(train_loader)
    Para.log_step = len(train_loader) // 4
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    t_loss, t_accu, v_loss, v_accu = [], [], [], []

    decay_cnt = 0
    #iterate through all epochs
    for epoch in range(1,  epoch_num + 1):
        if Para.cuda:
            model.cuda()
        #get training data
        train_loss, train_accuracy = train(model, epoch, train_loader, optimizer, loss_function)
        validation_loss, validation_accuracy = validate_test(model, epoch, use_loader=valid_loader, loss_function=loss_function)

        t_loss.append(train_loss)
        t_accu.append(train_accuracy)

        v_loss.append(validation_loss)
        v_accu.append(validation_accuracy)

        # use accuracy to find the best model
        if np.max(t_accu) == t_accu[-1]:
            print('***Found Best Training Model***')
        if np.max(v_accu) == v_accu[-1]:
            with open(Para.MODEL_SAVE_FOlD + save_name, 'wb') as f:
                torch.save(model.cpu().state_dict(), f)
                print('***Best Validation Model Found and Saved***')

        print('-' * 99)

        # Use BCE loss value for learning rate scheduling
        decay_cnt += 1

        if np.min(t_loss) not in t_loss[-3:] and decay_cnt > 2:
            scheduler.step()
            decay_cnt = 0
            print('***Learning rate decreased***')
            print('-' * 99)

        #determine callback
        callback(train_loss, validation_loss)
        if callback.stop == True:
          break;

    build_dict = {
        "train_loss": t_loss,
        "train_accu": t_accu,
        "valid_loss": v_loss,
        "valid_accu": v_accu,
    }

    with open(Para.LOG_SAVE_FOLD + log_name, 'w+') as lf:
        json.dump(build_dict, lf)

    return build_dict

#create confusion matrix
def record_matrix(model, use_loader, log_name):
  model = model.eval()
  data_loader_use = use_loader
  _index = 0
  result = list()
  for _index, data in enumerate(data_loader_use):
      spec_input, target = data['mel'], data['tag']

      if Para.cuda:
          spec_input = spec_input.cuda()
          target = target.cuda()

      with torch.no_grad():
          predicted = model(spec_input)
          _, predicted_index = torch.max(predicted, 1)
          _, target_index = torch.max(target, 1)
          m_tuple_list = [[int(predicted_index[i]), int(target_index[i])] for i in range(len(predicted_index))]
          result += m_tuple_list

  print('End of Matrix Record, Save file in {0}'.format(Para.LOG_SAVE_FOLD + log_name))
  print('-' * 99)
  with open(Para.LOG_SAVE_FOLD + log_name, 'w+') as f:
      json.dump(result, f)
  return
