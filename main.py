import torch
import os
import torch.nn as nn
from PIL import Image
from model import CNN_LSTM_Attention
import torchvision.transforms as transforms
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import csv
import cv2
import matplotlib.pyplot as plt
import time


def load_data(image_folder,  data_filename):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_list = []
    md_list = []

    with open(data_filename, 'r') as file:
        lines = file.readlines()

    for idx in range(1, len(lines) + 1):
        # Load images
        img_filename = os.path.join(image_folder, f"{idx:04d}.png")
        image = Image.open(img_filename).convert('RGB')
        input_tensor = preprocess(image)
        image_list.append(input_tensor)

        # Load multidimensional data
        data_line = lines[idx - 1]
        data_values = [float(val) for val in data_line.strip().split()]
        data_tensor = torch.tensor(data_values)
        md_list.append(data_tensor)

    image_tensor = torch.stack(image_list)
    md_tensor = torch.stack(md_list)

    return image_tensor, md_tensor


def train(net, image_data, md_data, optimizer, criterion, device, EPOCH):
    net.train()
    train_time_list = []
    train_loss_list = []  # Initialize an empty list to store training losses
    since = time.time()
    for j in range(EPOCH):

        target_filename = 'data/PMF_3_train.txt'
        target_list = []

        with open(target_filename, 'r') as file:
            lines = file.readlines()

        for line in lines:
            target_values = [float(val) for val in line.strip().split()]
            target_tensor = torch.tensor(target_values)
            target_list.append(target_tensor)

            target_tensor = torch.stack(target_list).to(device)

        for i in range(len(target_list)):
            """加载数据"""
            image_data_single = image_data[i].unsqueeze(0).to(device)
            md_data_single = md_data[i].unsqueeze(0).to(device)

            # 初始化梯度为 0
            optimizer.zero_grad()
            # 将数据喂给网络
            output,attention_map = net(image_data_single, md_data_single)
            losses = criterion(output, target_tensor[i].unsqueeze(0))
            losses.backward()  # backpropagation, compute gradients
            optimizer.step()  # 参数优化

            # Append the loss value to the list
            train_loss_list.append(losses.item())

            time_elapsed = time.time() - since
            print('Training complete in ECHPE{:.0f} {:.0f}m {:.0f}s'.format(
               j, time_elapsed // 60, time_elapsed % 60))
            train_time_list.append(time_elapsed)
    return train_loss_list, train_time_list


def test(net, image_data, md_data, criterion, device):
    net.eval()

    target_filename = 'data/PMF_4.txt'
    target_list = []
    test_platy_outputs = []
    test_lstm_outputs = []
    test_fix_outputs = []

    with open(target_filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        target_values = [float(val) for val in line.strip().split()]
        target_tensor = torch.tensor(target_values)
        target_list.append(target_tensor)

    target_tensor = torch.stack(target_list).to(device)

    with torch.no_grad():  # 不会计算梯度，也不会进行反向传播F
        for i in range(len(target_list)):
            """加载数据"""
            image_data_single = image_data[i].unsqueeze(0).to(device)
            md_data_single = md_data[i].unsqueeze(0).to(device)
            if i == 1000 and i == 2000 and i == 3000:
                print(i)
                image_path = 'data/img1/{:.0f}.png'.format(i)
                img = cv2.imread(image_path, 1)  # 用cv2加载原始图像
                output, attention_map = net(image_data_single, md_data_single)
                heatmap = attention_map.detach().cpu().numpy()
                heatmap = heatmap.reshape((3, 10, 10))
                heatmap = np.sum(heatmap, axis=0)
                min_val = np.min(heatmap)
                max_val = np.max(heatmap)
                heatmap = (heatmap - min_val) / (max_val - min_val)
                plt.matshow(heatmap)
                plt.savefig('heatmap-{:.0f}-cnn2.jpg'.format(i))
                plt.close()
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + img
                cv2.imwrite('cam{:.0f}-cnn2.jpg'.format(i), superimposed_img)
            else:
                output_platy, attention_map = net(image_data_single, md_data_single)
            output_platy_tensor = output_platy.clone().detach()
            test_platy_outputs.append(output_platy_tensor)
    test_platy_outputs_tensor = np.reshape(torch.stack(test_platy_outputs).to(device), (3293, 1))

    return target_tensor, test_platy_outputs_tensor


def percentage_within_threshold(pred, target, threshold):
    diff = torch.abs(pred - target)
    correct_predictions = torch.sum(diff <= threshold)
    return correct_predictions.item() / len(target)


def mean_squared_error(pred, target):
    return torch.mean((pred - target) ** 2)


def root_mean_squared_error(pred, target):
    return torch.sqrt(mean_squared_error(pred, target))


def mean_absolute_error(pred, target):
    return torch.mean(torch.abs(pred - target))


def median_absolute_error(pred, target):
    return torch.median(torch.abs(pred - target))


def mean_absolute_percentage_error(pred, target):
    return torch.mean(torch.abs((target - pred) / (target+0.01))) * 100


def explained_variance_score(pred, target):
    return metrics.explained_variance_score(target.cpu().numpy(), pred.cpu().numpy())


def r_squared(pred, target):
    y_bar = torch.mean(target)
    ss_tot = torch.sum((target - y_bar) ** 2)
    ss_res = torch.sum((pred - target) ** 2)
    return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    """读取数据"""

    # 加载训练集数据
    train_image_folder = 'data/img_new/'
    train_md_file = 'data/data_3_train.txt'
    train_image_tensor, train_md_tensor = load_data(train_image_folder, train_md_file)

    # 加载测试集数据
    test_image_folder = 'data/pull_4/'
    test_md_file = 'data/data_4.txt'
    test_image_tensor, test_md_tensor = load_data(test_image_folder, test_md_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """加载网络"""
    EPOCH = 1   # 定义迭代轮次
    learningRate = 0.0005
    criterion = nn.MSELoss()
    net = CNN_LSTM_Attention().to(device)
    optimizer = torch.optim.NAdam(net.parameters(), lr=learningRate)

    train_loss_list, train_time_list = train(net, train_image_tensor, train_md_tensor, optimizer, criterion, device, EPOCH)
    target_tensor, test_outputs = test(net, test_image_tensor, test_md_tensor, criterion, device)


    # Assuming test_outputs and target_tensor are already on the same device (GPU or CPU)
    test_outputs_numpy = test_outputs.cpu().numpy()
    target_tensor_numpy = target_tensor.cpu().numpy()

    # Convert numpy arrays to DataFrames
    df_test_outputs = pd.DataFrame(test_outputs_numpy,
                                    columns=[f"prediction_{i}" for i in range(test_outputs_numpy.shape[1])])
    df_target_tensor = pd.DataFrame(target_tensor_numpy,
                                    columns=[f"target_{i}" for i in range(target_tensor_numpy.shape[1])])

    # Concatenate DataFrames along columns
    df_combined = pd.concat([df_test_outputs, df_target_tensor], axis=1)

    # Save the combined DataFrame to a CSV file
    csv_filename = "test_results.csv"
    df_combined.to_csv(csv_filename, index=False)

    loss_filename = "train_loss.csv"  # CSV文件的文件名
    with open(loss_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            # 写入标题行（如果需要的话）
            # csv_writer.writerow(["Train Loss"])

            # 将整个训练损失值列表写入CSV文件的多行
            csv_writer.writerows([[losses] for losses in train_loss_list])

    times_filename = "train_times.csv"  # CSV文件的文件名
    with open(times_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        # 写入标题行（如果需要的话）
        # csv_writer.writerow(["Train Loss"])

        # 将整个训练损失值列表写入CSV文件的多行
        csv_writer.writerows([[time_elapsed] for time_elapsed in train_time_list])

    max_value = torch.max(target_tensor)
    min_value = torch.min(target_tensor)
    difference = max_value - min_value
    sqrt_difference = torch.sqrt(difference)
    sqrt_difference_2 = torch.sqrt(sqrt_difference)

    test_outputs = test_outputs.to(device)
    target_tensor = target_tensor.to(device)

    r2 = r_squared(test_outputs, target_tensor)
    mse = mean_squared_error(test_outputs, target_tensor)
    rmse = root_mean_squared_error(test_outputs, target_tensor)
    mae = mean_absolute_error(test_outputs, target_tensor)
    medae = median_absolute_error(test_outputs, target_tensor)
    mape = mean_absolute_percentage_error(test_outputs, target_tensor)
    ev = explained_variance_score(test_outputs, target_tensor)
    accuracy_threshold = sqrt_difference-sqrt_difference_2  # Set the threshold for accuracy evaluation
    # accuracy_threshold = 10  # Set the threshold for accuracy evaluation
    accuracy = percentage_within_threshold(test_outputs, target_tensor, accuracy_threshold)

    print(f"Mean Squared Error (MSE): {mse.item()}")
    print(f"Root Mean Squared Error (RMSE): {rmse.item()}")
    print(f"Mean Absolute Error (MAE): {mae.item()}")
    print(f"Median Absolute Error (MedAE): {medae.item()}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape.item()}%")
    print(f"Explained Variance Score (EV): {ev.item()}")
    print(f"R-squared (R^2) score: {r2.item()}")
    print(f"Accuracy (within {accuracy_threshold}): {accuracy * 100:.2f}%")

    # Create a list of dictionaries containing your data
    result_data = [
        {
            'Metric': 'R-squared',
            'Value': r2
        },
        {
            'Metric': 'Mean Squared Error',
            'Value': mse
        },
        {
            'Metric': 'Root Mean Squared Error',
            'Value': rmse
        },
        {
            'Metric': 'Mean Absolute Error',
            'Value': mae
        },
        {
            'Metric': 'Median Absolute Error',
            'Value': medae
        },
        {
            'Metric': 'Mean Absolute Percentage Error',
            'Value': mape
        },
        {
            'Metric': 'Explained Variance',
            'Value': ev
        },
        {
            'Metric': 'Accuracy Threshold',
            'Value': accuracy_threshold
        },
        {
            'Metric': 'Accuracy',
            'Value': accuracy
        }
    ]

    # Specify the CSV file path
    csv_file = 'metrics.csv'

    # Write data to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Metric', 'Value'])
        writer.writeheader()
        writer.writerows(result_data)

    print(f'Data saved to {csv_file}')