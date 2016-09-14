def prob_mass_function(data, bins='auto', range_min=None, range_max=None):
    
    import numpy as np
    
    data = data.flatten()
    
    if range_min is None:
        range_min = np.nanmin(data)
    if range_max is None:
        range_max = np.nanmax(data)
        
    if bins == 'auto':
        bins = data.shape[0]/10
        
    p_mass, x_values = np.histogram(data, bins=bins, range=(range_min, range_max))
    p_mass = p_mass / data.shape[0]
    x_values = x_values[:-1] + (x_values[1] - x_values[0])/2
    
    return p_mass, x_values




def jensenshannon(data, range_min=None, range_max=None, base=2, weights=None):
    
    import numpy as np
    import scipy as sp
    
    if range_min is None:
        data_mins = [np.nanmin(m) for m in data]
        range_min = data_mins[np.argmin(data_mins)]
    
    if range_max is None:
        data_maxs = [np.nanmax(n) for n in data]
        range_max = data_maxs[np.argmax(data_maxs)]
    
    data_shapes = [k.shape[0] for k in data]
    bins = int(np.round(data_shapes[np.argmin(data_shapes)]/10))
    p_mass = np.zeros((len(data), bins))
    
    combined_data = []
    for i in range(len(data)):
        p_mass[i], _ = prob_mass_function(data[i], bins, range_min, range_max)
        combined_data += list(data[i])
        
    p_mass_combined, _ = prob_mass_function(np.asarray(combined_data), bins, range_min, range_max)
    
    shannon_entropy = [sp.stats.entropy(p_mass[j], base=base) for j in range(p_mass.shape[0])]
    shared_shannon_entropy = sp.stats.entropy(p_mass_combined, base=base)
    
    if weights is None:
        weights = 1/len(data)
        
    shannon_entropy = np.asarray(shannon_entropy) * np.asarray(weights)
    
    jsdivergence = shared_shannon_entropy - np.sum(shannon_entropy)
    
    jsdistance = np.sqrt(jsdivergence)
    
    return jsdivergence, jsdistance