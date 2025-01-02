import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image


# Функция для ввода и предобработки данных
def my_input_and_processing():

    # Поиск всех изображений в каталоге и запись полученных
    # данных в датафрейм
    p = Path('data/images')
    list_name_img = []
    list_var_arr = []
    for x in p.glob("*"):
        x_rprefix = str(x).removeprefix('data\images\\')
        x_rprefix = x_rprefix[:10]
        list_name_img.append(x_rprefix)

        # Сжатие изображений до размера 32 х 32 и 
        # запись данных изображения в датафрейм
        img = Image.open(x)
        img_resize = img.resize((32, 32))
        list_var_arr.append(np.array(img_resize) / 255.0)
    list_name_img_df = pd.DataFrame({'id': list_name_img, 'val_arr': list_var_arr})

    # Формирование датафрейма из данных о тренировочной выборке
    train_df = pd.read_csv('data/train.csv')
    train_df_id = train_df['id'].str[:10]
    train_df_target = train_df['target']
    train_df_new = pd.concat([train_df_id, train_df_target], axis=1)

    # Соединение датафреймов с данными о тренировочной выборке и 
    # данными о изображении 
    result_train_df = pd.merge(train_df_new, list_name_img_df, on='id', how='left')

    # Формирование датафрейма из данных о тестовой выборке
    test_df = pd.read_csv('data/test.csv')
    test_images_df = pd.DataFrame(test_df['id'].str[:10])

    # Соединение датафреймов с данными о тестовой выборке и 
    # данными о изображении 
    result_test_df = pd.merge(test_images_df, list_name_img_df, on='id', how='left')


    # Формирование numpy-массивов данных о изображении
    all_images = result_train_df['val_arr'].to_numpy()
    train_images = all_images[:800]
    test_images = all_images[800:]

    all_labels = result_train_df['target'].to_numpy()
    train_labels = all_labels[:800]
    test_labels = all_labels[800:]

    t_images = result_test_df['val_arr'].to_numpy()


    # Преобразование numpy-массивов для корректной работы с 
    # Tensorflow
    train_img = np.zeros([train_images.shape[0], train_images[0].shape[0], train_images[0].shape[1], train_images[0].shape[2] + 1], dtype=np.float32)

    for i in range(train_images.shape[0]):
        if train_images[i].shape[2] == 3:
            train_img[i, :, :, :3] = train_images[i]
        else:
            train_img[i, :, :, :4] = train_images[i]


    test_img = np.zeros([test_images.shape[0], test_images[0].shape[0], test_images[0].shape[1], test_images[0].shape[2] + 1], dtype=np.float32)

    for i in range(test_images.shape[0]):
        if test_images[i].shape[2] == 3:
            test_img[i, :, :, :3] = test_images[i]
        else:
            test_img[i, :, :, :4] = test_images[i]

    t_img = np.zeros([t_images.shape[0], t_images[0].shape[0], t_images[0].shape[1], t_images[0].shape[2] + 1], dtype=np.float32)

    for i in range(t_images.shape[0]):
        if t_images[i].shape[2] == 3:
            t_img[i, :, :, :3] = t_images[i]
        else:
            t_img[i, :, :, :4] = t_images[i]

    # Вывод тренировочных, валидационных и тестовых данных, а также тестовый датафрейм 
    # для функции вывода
    return (train_img, train_labels), (test_img, test_labels), t_img, result_test_df


# Функция для пост-обработки и формирования файла результатов
def my_output_processing(result_test_df, predictions, num):

    # Формирование датафрейма для дальнейшего формаирования
    # верного файла ответов
    output = result_test_df
    output = output.drop(['val_arr'], axis=1)

    # Запись предсказаний вероятности того, что изображение
    # сгенерировано
    pred = []
    for i in range(predictions.shape[0]):
        pred.append(predictions[i][1])

    # Формирование файла результатов, который не соответствует 
    # нужному шаблону
    res_target = pd.DataFrame({'target': pred})
    res = output.join(res_target)
    name_sub = 'output/submission_' + str(num) + '.csv'
    res.to_csv(name_sub, index=False)
    print('Submission successfully saved')

    # Формирование файла результатов, который имеет
    # верную структуру 
    crutch_test = pd.read_csv('data/test.csv')
    crutch_sub = pd.read_csv(name_sub)
    crutch_sub = crutch_sub.rename(columns={'id':'id_res'})
    crutch_res = crutch_test.join(crutch_sub)
    crutch_res = crutch_res.drop({'id_res'}, axis=1)
    name_my_sub = 'output/my_submission_' + str(num) + '.csv'
    crutch_res.to_csv(name_my_sub, index=False)
    print('Submission successfully saved')
