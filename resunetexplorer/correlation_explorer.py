from resunetexplorer.utils import feature_maps_interp, image_sampling, binary_dilation, feature_maps_masking, get_image_label
import pandas as pd
import numpy as np
import pyprog
import torch
import gc # Garbage colector
import json # DataFrame.to_json



class CorrelationExplorer:


  # TODO: comentários sobre a ordem dos acontecimentos. A razão de dilatar as labels, 
  # a ordem de dilatar antes de sampling, 
  def pre_processing(self, layers, layers_fm_list, label, iterations_level = 7):

    masked_fm_dict = {}
    pair_fm_list = []
    

    i = 0

    while i <  len(layers_fm_list) - 1: 
      
      # Copy feature maps sets of position i and i+1 to auxiliar list
      pair_fm_list.append(layers_fm_list[i])
      pair_fm_list.append(layers_fm_list[i+1])

      # Test if the feature maps size of both layers are different 
      # and compute the scale factor between both feature maps.    
      if pair_fm_list[0].shape[1] > pair_fm_list[1].shape[1]:
        # Compute scale_factor
        scale_factor = pair_fm_list[0].shape[1]/pair_fm_list[1].shape[1]
        # Set index to the biggest and smallest one
        bigger_fm_idx = 0
        smaller_fm_idx = 1

      else:
        scale_factor = pair_fm_list[1].shape[1]/pair_fm_list[0].shape[1]
        bigger_fm_idx = 1
        smaller_fm_idx = 0


      # Apply interpolation over the smallest feature maps 
      feature_maps_interpolated = feature_maps_interp(pair_fm_list[smaller_fm_idx], scale_factor_ =  scale_factor)
      # Overwrite with interpolated feature maps with 
      pair_fm_list[smaller_fm_idx] = feature_maps_interpolated   
 
      # Apply binary dilation over image label
      dilated_label = binary_dilation(label, iterations_level = iterations_level)

      
      new_label = dilated_label

      # Test if the dilated label size is bigger than feature maps size 
      # and compute the scale factor between label and feature maps. 
      if dilated_label.shape[0] > pair_fm_list[bigger_fm_idx].shape[1]:
        scale_factor_2 =  dilated_label.shape[0]//pair_fm_list[bigger_fm_idx].shape[1]
        new_label = image_sampling(dilated_label, scale_factor_2)
        
      
      # Apply masking over all feature maps
      masked_fm_list = []
      for layer, layer_fm in enumerate(pair_fm_list):
        masked_fm  = feature_maps_masking(layer_fm, new_label)
        masked_fm_list.append(masked_fm)

      # Store masked feature maps into respective dictionary key
      masked_fm_dict[f"{layers['layer_path'][i]}-{layers['layer_path'][i+1]}"] = masked_fm_list

      # Clear memory
      del feature_maps_interpolated
      del pair_fm_list
      del masked_fm_list
      del masked_fm
      gc.collect()

      pair_fm_list = []

      # Increase iterator
      i = i + 1


    return masked_fm_dict

  # TODO: documentar a função
  def feature_maps_correlation(self, layer_1_fm, layer_2_fm, layer_1_name, layer_2_name, n_maps1, n_maps2):

  
    fm_correlation = pd.DataFrame(columns=(layer_1_name+'_fm_id', layer_2_name+'_fm_id', 'correlation'))

    # Create Object to progress bar
    prog = pyprog.ProgressBar(" ", "", n_maps1)
    # Print Task name
    print(f'Computing feature maps correlation: {layer_1_name} - {layer_2_name} \n')
    # Update Progress Bar
    prog.update()


    for map_idx1 in range(n_maps1):
      # Reshaping it into a one-dimensional tensor
      layer_1_map_1d = layer_1_fm[map_idx1].flatten()

      for map_idx2 in range(n_maps2):   
        # Ceshaping it into a one-dimensional tensor
        layer_2_map_1d = layer_2_fm[map_idx2].flatten()
        # Concatenate two tensors along a new dimension.
        x = torch.stack([layer_1_map_1d, layer_2_map_1d])
        # Move tensor from CPU to GPU
        if torch.cuda.is_available():
          x = x.cuda()
        # Get correlation between two feature maps
        corr = torch.corrcoef(x)[0][1]
        # Check if tensor is on GPU
        if corr.is_cuda:
           # Move tensor from GPU to CPU and transform to NumPy
          corr = corr.cpu().detach().numpy()
        else:
          corr.numpy()
          
        # Append data to dict
        fm_correlation_dict = {
            layer_1_name+'_fm_id' : map_idx1, 
            layer_2_name+'_fm_id' : map_idx2,
            'correlation'         : corr

        }

        # Convert the dict to dataframe
        df_result = pd.DataFrame([fm_correlation_dict])
        # Concact df_result with similarity_metrics to append new row
        fm_correlation = pd.concat([fm_correlation, df_result])
        
      # Set current status
      prog.set_stat(map_idx1 + 1)
      # Update Progress Bar
      prog.update()

    # Make the Progress Bar final
    prog.end()
    print('\n')

    # Reset indexes
    fm_correlation.reset_index(drop = True, inplace = True)

    return fm_correlation
  
  # TODO: documentar a função
  def corr_calculation(self, layers, masked_fm_dict):
    

    # A dict to store DataFrames of feature maps correlations 
    fm_corr_dict = {}

    for i, key in enumerate(masked_fm_dict.keys()): 
      
      layer_1_name = layers['layer_path'][i]
      layer_2_name = layers['layer_path'][i+1]
      n_maps1 = layers['n_maps'][i]
      n_maps2 = layers['n_maps'][i+1]

      fm_corr_dict[f'corr_{key}'] = self.feature_maps_correlation(masked_fm_dict[key][0], masked_fm_dict[key][1], layer_1_name, layer_2_name, n_maps1, n_maps2)    
       
    return fm_corr_dict

  # TODO: documentar a função
  def get_max_correlations(self, layers, fm_corr_dict):
    """
      Get the max correlation between two differents 
    """

    fm_corr_max_dict = {}

    for i, key in enumerate(fm_corr_dict.keys()): 

      if layers['n_maps'][i] > layers['n_maps'][i+1]:
        column_groupby =  fm_corr_dict[key].columns[0]
      else:
        column_groupby =  fm_corr_dict[key].columns[1]

      fm_corr_max_dict[f'max_{key}'] = fm_corr_dict[key].loc[fm_corr_dict[key].astype(float).groupby(column_groupby)['correlation'].idxmax()].reset_index(drop = True)

    return fm_corr_max_dict


  # TODO: documentar a função
  def get_correlation_freq(self, layers, fm_corr_max_dict):

    """From the maximum correlation, this function calculates the frequency of correlation occurrence 
    of feature maps, their mean, median and standard deviation.      
    """
    stats_most_freq_corr = {}

    for i, key in enumerate(fm_corr_max_dict.keys()): 

      if layers['n_maps'][0] < layers['n_maps'][1]:
        column =  fm_corr_max_dict[key].columns[0]
      else:
        column =  fm_corr_max_dict[key].columns[1]

      print(f'column: {column}')
      print(f'key: {key}')

      aux = fm_corr_max_dict[key].copy()
      aux[fm_corr_max_dict[key].columns[0]] = aux[fm_corr_max_dict[key].columns[0]].astype('str')
      aux[fm_corr_max_dict[key].columns[1]] = aux[fm_corr_max_dict[key].columns[1]].astype('str')
      stats_most_freq_corr[f'freq_{key}'] = pd.DataFrame(aux[column].value_counts())
      # Reset index
      stats_most_freq_corr[f'freq_{key}'] = stats_most_freq_corr[f'freq_{key}'].reset_index()
      # Rename columns
      stats_most_freq_corr[f'freq_{key}'].columns = [column, 'counts']
      # Calculate relative frequency
      stats_most_freq_corr[f'freq_{key}']['freq%'] = round(100*stats_most_freq_corr[f'freq_{key}']['counts']/stats_most_freq_corr[f'freq_{key}']['counts'].sum(), 4)
      # Calculate mean, median and std
      stats_most_freq_corr[f'freq_{key}']['mean'] = stats_most_freq_corr[f'freq_{key}'][column].astype('int64').apply(lambda x: fm_corr_max_dict[key][fm_corr_max_dict[key][column] == x]['correlation'].mean())
      stats_most_freq_corr[f'freq_{key}']['median'] = stats_most_freq_corr[f'freq_{key}'][column].astype('int64').apply(lambda x: fm_corr_max_dict[key][fm_corr_max_dict[key][column] == x]['correlation'].median())
      stats_most_freq_corr[f'freq_{key}']['std'] = stats_most_freq_corr[f'freq_{key}'][column].astype('int64').apply(lambda x: fm_corr_max_dict[key][fm_corr_max_dict[key][column] == x]['correlation'].std())
      stats_most_freq_corr[f'freq_{key}']['std'].replace(np.nan, 0, inplace = True)

    return stats_most_freq_corr

  # TODO: documentar a função
  def correlation_pipeline(self, img, img_label, img_idx, layers_paths, model, save_path = None, file_type = 'csv',  iterations_level = 7,  device = 'cuda'):
    '''
    '''
    # Initialize ExtractResUNetLayers class
    erl = ExtractResUNetLayers(model)
    # Extract layers from model
    layers = erl.get_layers(layers_paths)
    # Initialize ExtractResUNetMaps class
    erm = ExtractResUNetMaps(model, dataset = None, image = img, device = 'cuda')
    # Extract feature maps from layers 
    layers_fm_list = erm.get_multiple_feature_maps(layers['layer'])
    # Initialize CorrelationExplorer class
    cxp = CorrelationExplorer()
    # Run pre processing to mask the features maps with dilated image label 
    masked_fm_dict = cxp.pre_processing(layers, layers_fm_list, img_label, iterations_level = iterations_level)
    # Compute correlation between features maps of subsequents layers
    fm_correlation_dict = cxp.corr_calculation(layers, masked_fm_dict)
    # Compute the maximum correlation between features maps of subsequents layers
    fm_corr_max_dict = cxp.get_max_correlations(layers, fm_correlation_dict)
    # Compute statistics for the most frequently correlated features maps
    stats_most_freq_corr = cxp.get_correlation_freq(layers, fm_corr_max_dict)

    # TODO Verificar se o diretório é válido. 
    if save_path != None:
      # Save stats_most_freq_corr's dataframes as json or CSV
      if file_type == 'json':
        for i, key in enumerate(stats_most_freq_corr.keys()):
          stats_most_freq_corr[key].to_json(path_or_buf = f'{save_path}/{key}.json', orient = "index")
      elif file_type == 'csv':
        stats_most_freq_corr[key].to_csv(path_or_buf = f'{save_path}/{key}.csv', sep=',', index = False)


    return masked_fm_dict, fm_correlation_dict, fm_corr_max_dict, stats_most_freq_corr
