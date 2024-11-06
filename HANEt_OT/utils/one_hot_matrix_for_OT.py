import torch


def get_one_hot_true_label_and_true_trigger(data_instance, num_label):
    true_label = []
    trigger_word = []
    seq_len = (
        len(data_instance["piece_ids"]) + 1
    )  # because start_index of piece_ids is 1 instead of 0
    for i in range(len(data_instance["label"])):
        if data_instance["label"][i] != 0:
            true_label.append(data_instance["label"][i])
            trigger_word.append(data_instance["span"][i])

    set_label_in_one_sentence = set(true_label)
    true_one_hot_trigger_vector = torch.zeros(num_label)
    for i in set_label_in_one_sentence:
        true_one_hot_trigger_vector += torch.eye(num_label)[i]

    true_one_hot_label_vector = torch.zeros(seq_len)
    trigger = []
    for i in trigger_word:
        trigger.extend(i)

    set_trig = set(trigger)
    for i in set_trig:
        true_one_hot_label_vector += torch.eye(seq_len)[i]
    return true_one_hot_trigger_vector, true_one_hot_label_vector
