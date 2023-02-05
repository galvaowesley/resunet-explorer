from resunetexplorer.utils import feature_maps_interp, image_sampling, binary_dilation, feature_maps_masking, get_image_label, remove_prefix
from resunetexplorer.layer_extractor import ExtractResUNetLayers
from resunetexplorer.maps_extractor import ExtractResUNetMaps

import pandas as pd
import numpy as np
import pyprog
import torch
import gc # Garbage colector
import json # DataFrame.to_json


class CorrelationExplorer:

 
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
        # Reshaping it into a one-dimensional tensor
        layer_2_map_1d = layer_2_fm[map_idx2].flatten()
        
        # Calculate standard deviation of both feature maps
        std_fm_1 = layer_1_map_1d.std()
        std_fm_2 = layer_2_map_1d.std()

        # Treatment to avoid NaN correlations.
        if (std_fm_1 < 1e-10) and (std_fm_2 < 1e-10):
          corr = 1.0
          # Flag that indicates whether the feature map is mostly composed of zero values. If std_fm < 1e-10, then zero_flag_fm = 1.
          zero_flag_fm_1 = 1
          zero_flag_fm_2 = 1
        elif (std_fm_1 < 1e-10) and (std_fm_2 >= 1e-10):
          corr = 0.0
          zero_flag_fm_1 = 1
          zero_flag_fm_2 = 0
        elif (std_fm_1 >= 1e-10) and (std_fm_2 < 1e-10): 
          corr = 0.0
          zero_flag_fm_1 = 0
          zero_flag_fm_2 = 1
        else:
          zero_flag_fm_1 = 0
          zero_flag_fm_2 = 0
          # Concatenate two tensors along a new dimension.
          x = torch.stack([layer_1_map_1d, layer_2_map_1d])
          # Move tensor from CPU to GPU
          if torch.cuda.is_available():
            x = x.cuda()
          # Get correlation between two feature maps
          corr = torch.abs(torch.corrcoef(x)[0][1])
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
            'correlation'         : corr,
            layer_1_name+'_zero_flag': zero_flag_fm_1,
            layer_2_name+'_zero_flag': zero_flag_fm_2,

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
  
  def corr_calculation(
            self,
            model1_name, 
            model2_name, 
            layers_metadata1, 
            layers_metadata2, 
            feature_list_model1, 
            feature_list_model2
  ):
    
    # A dict to store DataFrames of feature maps correlations 
    fm_corr_dict = {}

    for idx1, layer_path1 in enumerate(layers_metadata1['layer_path']):

      layer1_name = model1_name + layer_path1
      n_maps1 = layers_metadata1['n_maps'][idx1]      

      for idx2, layer_path2 in enumerate(layers_metadata2['layer_path']):

        layer2_name = model2_name + layer_path2
        n_maps2 = layers_metadata2['n_maps'][idx2]


        fm_corr_dict[f'{model1_name}_{model2_name}_{layer_path1}_{layer_path2}'] = self.feature_maps_correlation(
                feature_list_model1[:][idx1], 
                feature_list_model2[:][idx2], 
                layer1_name, 
                layer2_name, 
                n_maps1, 
                n_maps2
        )    
    
    return fm_corr_dict

  # TODO: documentar a função
  def get_max_correlations(
                self, 
                layers_metadata1, 
                layers_metadata2, 
                fm_corr_dict
  ):
    """
      Get the max correlation between two differents 
    """

    fm_corr_max_dict = {}

    for i, key in enumerate(fm_corr_dict.keys()): 

      if layers_metadata1['n_maps'][0] >= layers_metadata2['n_maps'][0]:
        column_groupby =  fm_corr_dict[key].columns[0]
      else:
        column_groupby =  fm_corr_dict[key].columns[1]

      fm_corr_max_dict[f'max_{key}'] = fm_corr_dict[key].loc[fm_corr_dict[key].astype(float).groupby(column_groupby)['correlation'].idxmax()].reset_index(drop = True)

    return fm_corr_max_dict


  # TODO: documentar a função
  def get_correlation_stats(
                self, 
                layers_metadata1, 
                layers_metadata2, 
                fm_corr_max_dict
    ):

    """From the maximum correlation, this function calculates the stats of the most frequent correlations between 
    feature maps of two models. The stats are: the number of feature maps, their mean, median and standard deviation.      
    """
    stats_most_freq_corr = {}

    for i, key in enumerate(fm_corr_max_dict.keys()): 

      if layers_metadata1['n_maps'][0] <= layers_metadata2['n_maps'][0]:
        column =  fm_corr_max_dict[key].columns[0]
      else:
        column =  fm_corr_max_dict[key].columns[1]

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
  def correlation_pipeline(
                self, 
                img, 
                layers_list1,
                layers_list2, 
                model1,
                model2, 
                model1_name, 
                model2_name, 
                save_path = None, 
                file_type = 'csv',                 
                device = 'cuda', 
                memory_cleaning = False
    ):
    '''
    '''
    # Initialize ExtractResUNetLayers class
    erl_model1 = ExtractResUNetLayers(model1)
    erl_model2 = ExtractResUNetLayers(model2)
    
    # Get layers metadata from model
    layers_metadata_model1 = erl_model1.get_layers(layers_list1)
    layers_metadata_model2 = erl_model2.get_layers(layers_list2)    
    
    # Initialize ExtractResUNetMaps 
    erm_model1 = ExtractResUNetMaps(model1, dataset = None, image = img, device = device)
    erm_model2 = ExtractResUNetMaps(model2, dataset = None, image = img, device = device)
    
    # Extract feature maps from layers 
    fm_list_model1 = erm_model1.get_multiple_feature_maps(layers_metadata_model1['layer'])
    fm_list_model2 = erm_model2.get_multiple_feature_maps(layers_metadata_model2['layer'])
    
    # Compute correlation between features maps of two models 
    fm_corr_dict = self.corr_calculation(
        model1_name=model1_name, 
        model2_name=model2_name, 
        layers_metadata1=layers_metadata_model1, 
        layers_metadata2=layers_metadata_model2, 
        feature_list_model1=fm_list_model1, 
        feature_list_model2=fm_list_model2
    )
    # Compute the maximum correlation between features maps of subsequents layers
    fm_corr_max_dict = self.get_max_correlations(layers_metadata_model1, layers_metadata_model2, fm_corr_dict)
    # Compute statistics for the most frequently correlated features maps
    stats_most_freq_corr = self.get_correlation_stats(layers_metadata_model1, layers_metadata_model2, fm_corr_max_dict)

    # TODO Verificar se o diretório é válido. 
    if save_path != None:
      # Save stats_most_freq_corr's dataframes as json or CSV
      if file_type == 'json':
        for i, key in enumerate(stats_most_freq_corr.keys()):
          # Export fm_corr_dict
          file_name = remove_prefix(key, 'freq_max_')
          fm_corr_dict[file_name].to_json(path_or_buf = f'{save_path}/{file_name}.json', orient = "index")
          # Export fm_corr_max_dict
          file_name = remove_prefix(key, 'freq_')
          fm_corr_max_dict[file_name].to_json(path_or_buf = f'{save_path}/{file_name}.json', orient = "index")
          # Export stats_most_freq_corr
          stats_most_freq_corr[key].to_json(path_or_buf = f'{save_path}/{key}.json', orient = "index")
      elif file_type == 'csv':
        for i, key in enumerate(stats_most_freq_corr.keys()):
          # Export fm_corr_dict
          file_name = remove_prefix(key, 'freq_max_')
          fm_corr_dict[file_name].to_csv(path_or_buf = f'{save_path}/{file_name}.csv', sep=',', index = False)
          # Export fm_corr_max_dict
          file_name = remove_prefix(key, 'freq_')
          fm_corr_max_dict[file_name].to_csv(path_or_buf = f'{save_path}/{file_name}.csv', sep=',', index = False)
          # Export stats_most_freq_corr
          stats_most_freq_corr[key].to_csv(path_or_buf = f'{save_path}/{key}.csv', sep=',', index = False)
  
    # Memory cleaning
    if memory_cleaning == True:
      del layers_fm_list
      gc.collect()
      torch.cuda.empty_cache()
    else:
      return fm_corr_dict, fm_corr_max_dict, stats_most_freq_corr
    