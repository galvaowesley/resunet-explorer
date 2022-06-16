from resunetexplorer.utils import feature_maps_interp, image_sampling, binary_dilation, feature_maps_masking, get_image_label
import pandas as pd

class CorrelationExplorer:

  # TODO: comentários sobre a ordem dos acontecimentos. A razão de dilatar as labels, 
  # a ordem de dilatar antes de sampling, 
  def pre_processing(self, layers, layers_fm_list, label, iterations_level = 7):

    
    # Test if the feature maps size of both layers are different 
    # and compute the scale factor between both feature maps.    
    if layers_fm_list[0].shape[1] > layers_fm_list[1].shape[1]:
      # Compute scale_factor
      scale_factor = layers_fm_list[0].shape[1]/layers_fm_list[1].shape[1]
      # Set index to the biggest and smallest one
      bigger_fm_idx = 0
      smaller_fm_idx = 1
    else:
      scale_factor = layers_fm_list[1].shape[1]/layers_fm_list[0].shape[1]
      bigger_fm_idx = 1
      smaller_fm_idx = 0

    # Apply interpolation over the smallest feature maps 
    feature_maps_interpolated = feature_maps_interp(layers_fm_list[smaller_fm_idx], scale_factor_ =  scale_factor)
    # Overwrite with interpolated feature maps with 
    layers_fm_list[smaller_fm_idx] = feature_maps_interpolated

    # Apply binary dilation over image label
    dilated_label = binary_dilation(label, iterations_level = iterations_level)
    
    new_label = dilated_label
    # Test if the dilated label size is bigger than feature maps size 
    # and compute the scale factor between label and feature maps. 
    if dilated_label.shape[0] > layers_fm_list[bigger_fm_idx].shape[1]:
      scale_factor_2 =  dilated_label.shape[0]//layers_fm_list[bigger_fm_idx].shape[1]
      new_label = image_sampling(dilated_label, scale_factor_2)


    # Apply masking over all feature maps
    masked_fm_list = []
    for i, layer_fm in enumerate(layers_fm_list):
      masked_fm  = feature_maps_masking(layer_fm, new_label)
      masked_fm_list.append(masked_fm)
    
    return masked_fm_list

  # TODO: documentar a função
  def feature_maps_correlation(self, layer_1_fm, layer_2_fm, layer_1_name, layer_2_name, n_maps1, n_maps2):

  
    fm_correlation = pd.DataFrame(columns=(layer_1_name+'_fm_id', layer_2_name+'_fm_id', 'correlation'))

    # Create Object to progress bar
    prog = pyprog.ProgressBar(" ", "", n_maps1)
    # Print Task name
    print('Computing feature maps correlation: \n')
    # Update Progress Bar
    prog.update()


    for map_idx1 in range(n_maps1):
      layer_1_map_1d = layer_1_fm[map_idx1].flatten()

      for map_idx2 in range(n_maps2):   
        
        layer_2_map_1d = layer_2_fm[map_idx2].flatten()
        corr = np.corrcoef(layer_1_map_1d, layer_2_map_1d)[0][1]

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

    # Reset indexes
    fm_correlation.reset_index(drop = True, inplace = True)

    return fm_correlation
  
  # TODO: documentar a função
  def corr_calculation(self, layers, masked_fm_list):
    
    name1 = layers['layer_path'][0]
    name2 = layers['layer_path'][1]
    nmaps1 = layers['n_maps'][0]
    nmaps2 = layers['n_maps'][1]


    fm_correlation = self.feature_maps_correlation(masked_fm_list[0], masked_fm_list[1], name1, name2, nmaps1, nmaps2)

    return fm_correlation

  # TODO: documentar a função
  def get_max_correlations(self, layers, fm_correlation):
    """
      Get the max correlation between two differents 
    """

    if layers['n_maps'][0] > layers['n_maps'][1]:
      column_groupby =  fm_correlation.columns[0]
    else:
      column_groupby =  fm_correlation.columns[1]

    fm_correlation_max = fm_correlation.loc[fm_correlation.groupby(column_groupby)['correlation'].idxmax()].reset_index(drop = True)

    return fm_correlation_max

  # TODO: documentar a função
  def get_correlation_freq(self, layers, fm_correlation_max):

    """
      
    """

    if layers['n_maps'][0] < layers['n_maps'][1]:
      column =  fm_correlation_max.columns[0]
    else:
      column =  fm_correlation_max.columns[1]

    aux[fm_correlation_max.columns[0]] = aux[fm_correlation_max.columns[0]].astype('str')
    aux[fm_correlation_max.columns[1]] = aux[fm_correlation_max.columns[1]].astype('str')
    fm_corr_freq = pd.DataFrame(aux[column].value_counts())
    fm_corr_freq = fm_corr_freq.reset_index()
    fm_corr_freq.columns = [column, 'counts']
    fm_corr_freq['freq%'] = round(100*fm_corr_freq['counts']/fm_corr_freq['counts'].sum(), 3)

    return fm_corr_freq
