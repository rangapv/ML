#This is the logistic regression using PyTorch for Bianry Classification
import torch as T
import numpy as np

class PatientDataset(T.utils.data.Dataset):
  def __init__(self, src_file, num_rows=None):
    all_data = np.loadtxt(src_file, max_rows=num_rows,
      usecols=range(0,9), delimiter="\t", skiprows=0,
      comments="#", dtype=np.float32)  # read all 9 columns

    self.x_data = T.tensor(all_data[:,1:9],
      dtype=T.float32).to(device)
    self.y_data = T.tensor(all_data[:,0],
      dtype=T.float32).to(device)

    self.y_data = self.y_data.reshape(-1,1)  # 2D

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    preds = self.x_data[idx,:]  # idx rows, all 8 cols
    sex = self.y_data[idx,:]    # idx rows, the only col
    sample = { 'predictors' : preds, 'sex' : sex }
    return sample

# ---------------------------------------------------------

class LogisticReg(T.nn.Module):
  def __init__(self):
    super(LogisticReg, self).__init__()
    self.fc = T.nn.Linear(8, 1)

    T.nn.init.uniform_(self.fc.weight, -0.01, 0.01)
    T.nn.init.zeros_(self.fc.bias)

  def forward(self, x):
    z = self.fc(x)
    p = T.sigmoid(z)
    return p

# ----------------------------------------------------------

def train(log_reg, ds, mi):
  log_reg.train()
  # model, dataset, batch size, max iterations
  # batch size must equal all data for L-BFGS
  # learn_rate not applicable
  # max_epochs is really max iterations
  # log_every isn't needed -- typically very few iterations

  # log_reg.train()  # set training mode not relevant
  # loss_func = T.nn.MSELoss()  # mean squared error - NO
  loss_func = T.nn.BCELoss()  # binary cross entropy
  # opt = T.optim.SGD(log_reg.parameters(), lr=lr)
  opt = T.optim.LBFGS(log_reg.parameters(), max_iter=mi)
  train_ldr = T.utils.data.DataLoader(ds,
    batch_size=len(ds), shuffle=False)  # shuffle irrelevant

  print("\nStarting L-BFGS training")

  for itr in range(0, mi):
    itr_loss = 0.0            # for one iteration
    for (_, all_data) in enumerate(train_ldr):  # b_ix irrelevant
      X = all_data['predictors']  # [10,8]  inputs
      Y = all_data['sex']         # [10,1]  targets

      # -------------------------------------------
      def loss_closure():
        opt.zero_grad()
        oupt = log_reg(X)
        loss_val = loss_func(oupt, Y)
        loss_val.backward()
        return loss_val
      # -------------------------------------------

      opt.step(loss_closure)  # get loss, use to update wts

      oupt = log_reg(X)  # monitor loss
      loss_val = loss_closure()
      itr_loss += loss_val.item()
    print("iteration = %4d   loss = %0.4f" % (itr, itr_loss))

  print("Done ")

# ----------------------------------------------------------

def accuracy(model, ds, verbose=False):
  # ds is a iterable Dataset of Tensors
  model.eval()
  n_correct = 0; n_wrong = 0

  for i in range(len(ds)):
    inpts = ds[i]['predictors']
    target = ds[i]['sex']    # float32  [0.0] or [1.0]
    with T.no_grad():
      oupt = model(inpts)
    if verbose == True:
      print("")
      print(oupt)
      print(target)
      input()

    # avoid 'target == 1.0'
    if target < 0.5 and oupt < 0.5:  # .item() not needed
      n_correct += 1
    elif target >= 0.5 and oupt >= 0.5:
      n_correct += 1
    else:
      n_wrong += 1

  return (n_correct * 1.0) / (n_correct + n_wrong)

# ----------------------------------------------------------

def main():
  # 0. get started
  print("\nPatient gender logisitic regression L-BFGS PyTorch ")
  print("Predict gender from age, county, monocyte, history")
  T.manual_seed(1)
  np.random.seed(1)

  # 1. create Dataset and DataLoader objects
  print("\nCreating Patient train and test Datasets ")

  train_file = "./patients_train.txt"
  test_file = "./patients_test.txt"

  train_ds = PatientDataset(train_file)  # read all rows
  test_ds = PatientDataset(test_file)

  # 2. create model
  print("\nCreating 8-1 logistic regression model ")
  log_reg = LogisticReg().to(device)

  # 3. train network
  print("\nPreparing L-BFGS training")
  # bat_size = len(train_ds)  # use all
  # lrn_rate = 0.10
  max_iterations = 4
  # ep_log_interval = 200
  print("Loss function: BCELoss ")
  print("Optimizer: L-BFGS ")
  # print("Learn rate: " + str(lrn_rate))
  # print("Batch size: " + str(bat_size))
  # print("Max epochs: " + str(max_epochs))

  # train(log_reg, train_ds, bat_size, lrn_rate, max_epochs,
  #   ep_log_interval)
  train(log_reg, train_ds, max_iterations)

# ----------------------------------------------------------

  # 4. evaluate model
  acc_train = accuracy(log_reg, train_ds)
  print("\nAccuracy on train data = %0.2f%%" % \
    (acc_train * 100))
  acc_test = accuracy(log_reg, test_ds, verbose=False)
  print("Accuracy on test data = %0.2f%%" % \
    (acc_test * 100))

  # 5. examine model
  wts = log_reg.fc.weight.data
  print("\nModel weights: ")
  print(wts)
  bias = log_reg.fc.bias.data
  print("Model bias: ")
  print(bias)

  # 6. save model
  # print("Saving trained model state_dict \n")
  # path = ".\\Models\\patients_LR_model.pth"
  # T.save(log_reg.state_dict(), path)

  # 7. make a prediction
  print("Predicting sex for age = 30, county = carson, ")
  print("monocyte count = 0.4000, ")
  print("hospitization history = moderate ")
  inpt = np.array([[0.30, 0,0,1, 0.40, 0,1,0]],
    dtype=np.float32)
  inpt = T.tensor(inpt, dtype=T.float32).to(device)

  log_reg.eval()
  with T.no_grad():
    oupt = log_reg(inpt)    # a Tensor
  pred_prob = oupt.item()   # scalar, [0.0, 1.0]
  print("\nComputed output pp: ", end="")
  print("%0.4f" % pred_prob)

  if pred_prob < 0.5:
    print("Prediction = male")
  else:
    print("Prediction = female")

  print("\nEnd Patient gender demo")

if __name__== "__main__":
  main()
