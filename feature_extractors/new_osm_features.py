def make_conj(data, feature1, feature2):
    data[feature1 + ' + ' + feature2] = data[feature1].astype(str) + ' + ' + data[feature2].astype(str)
    return (data)


def make_product(data, feature1, feature2):
    """Вычисление произведения расстояния до метро и transport_stop(или других фичей, но это кажется круто)
        example: make_product(df_with_new_features, 'subway_dist', 'transport_stop_closest_dist')
    """
    data['subway+transport'] = data[[feature1, feature2]].product(axis=1)
    return (data)


def make_sum(data, feature1, feature2):
    data['sum_features'] = data[[feature1, feature2]].sum(axis=1)
    return (data)


def make_sub(data, feature1, feature2):
    data['sub_features'] = data[[feature1, feature2]].sub(axis=1)
    return (data)


def make_div(data, feature1, feature2):
    data['div_features'] = data[[feature1, feature2]].div(axis=1)
    return (data)
