import pandas as pd


def get_all_click_data(mode):
    train_data_path = "Datasets/train.csv"
    test_data_path = "Datasets/test.csv"
    train_data = pd.read_csv(
        train_data_path,
        sep=',',
        encoding='utf-8'
    )
    test_data = pd.read_csv(
        test_data_path,
        sep=',',
        encoding='utf-8'
    )
    if mode == "offline":
        return train_data
    elif mode == "online":
        all_click = train_data.append(test_data)
        all_click = all_click.drop_duplicates(['user_id', 'item_id', 'timestamp'], keep='last')
        return all_click, test_data


def get_answer():
    answer_path = "Datasets/answer.csv"
    answer = pd.read_csv(
        answer_path,
        sep=',',
        encoding="utf-8"
    )
    return answer


