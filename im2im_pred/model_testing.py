from fastai.imports import torch, np, Path, os
from tqdm import tqdm

import architectures
from create_dataset import NYUv2


def evaluate_model(model, test_loader, device, index):
    # evaluating test data
    with torch.no_grad():  # operations inside don't track history

        avg_cost = np.zeros(12, dtype=np.float32)

        for test_data, test_label, test_depth, test_normal in tqdm(test_loader, desc='Testing'):
            test_data, test_label = test_data.to(device), test_label.type(torch.LongTensor).to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            test_pred, _ = model(test_data)
            test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

            cost = np.zeros(12, dtype=np.float32)

            cost[0] = test_loss[0].item()
            cost[1] = model.compute_miou(test_pred[0], test_label).item()
            cost[2] = model.compute_iou(test_pred[0], test_label).item()
            cost[3] = test_loss[1].item()
            cost[4], cost[5] = model.depth_error(test_pred[1], test_depth)
            cost[6] = test_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(test_pred[2], test_normal)

            # TODO probably this averaging is not okay for these metrics
            avg_cost += cost / len(test_loader)

        performance = '''Epoch: {:04d} | TEST : {:.4f} | {:.4f} {:.4f} | {:.4f} | {:.4f} {:.4f} | {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'''.format(
            index,
            avg_cost[0], avg_cost[1],
            avg_cost[2], avg_cost[3], avg_cost[4], avg_cost[5],
            avg_cost[6],
            avg_cost[7], avg_cost[8], avg_cost[9], avg_cost[10],
            avg_cost[11])
        return avg_cost, performance


def load_model(CHECKPOINT_PATH, ModelClass, device, **kwargs):
    model = ModelClass(device, **kwargs)

    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_saved_model(CHECKPOINT_PATH, ModelClass, device, **kwargs):
    dataset_path = 'data/nyuv2'
    nyuv2_test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 2
    num_workers = 2

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    model = load_model(CHECKPOINT_PATH, ModelClass, device, **kwargs)
    return evaluate_model(model, nyuv2_test_loader, device, -1)


def write_performance(name_model_run, performance, loss_str):
    PERFORMANCE_PATH = Path('./logs/{}/'.format(name_model_run))
    os.makedirs(PERFORMANCE_PATH, exist_ok=True)
    with open(PERFORMANCE_PATH / 'final_performance.txt', 'w') as handle:
        handle.write(loss_str)
        handle.write(performance)


if __name__ == '__main__':
    loss_str = 'LOSS FORMAT: SEMANTIC_LOSS | MEAN_IOU PIX_ACC | DEPTH_LOSS | ABS_ERR REL_ERR | NORMAL_LOSS | MEAN MED <11.25 <22.5 <30\n'

    name_model_run = 'mtan_segnet_without_attention_dwa_run_0'
    device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")

    CHECKPOINT_PATH = Path('./logs/{}/model_checkpoints/checkpoint.chk'.format(name_model_run))

    avg_cost, performance = evaluate_saved_model(CHECKPOINT_PATH, architectures.SegNetWithoutAttention, device)

    write_performance(name_model_run, performance, loss_str)
