from resunetexplorer.utils import feature_maps_interp, image_sampling, binary_dilation, feature_maps_masking, \
    get_image_label, remove_prefix
from resunetexplorer.layer_extractor import ExtractResUNetLayers
from resunetexplorer.maps_extractor import ExtractResUNetMaps

import pandas as pd
import numpy as np
import pyprog
import torch
import gc  # Garbage colector
import json  # DataFrame.to_json


def check_models_name(model1_name, model2_name):
    """Function to check if the names of the models are the same.
    """
    if model1_name == model2_name:
        model1_name = model1_name + '(1)'
        model2_name = model2_name + '(2)'
    else:
        model1_name
        model2_name

    return model1_name, model2_name


class CorrelationExplorer:

    def check_if_map_is_zero(
            self,
            feature_map1,
            feature_map2,
    ):
        """Function to check if the pair of feature maps are mostly composed of zero values. It's a treatment to avoid
      NaN correlations.
      """
        # Calculate standard deviation of both feature maps
        std_fm_1 = feature_map1.std()
        std_fm_2 = feature_map2.std()

        if (std_fm_1 < 1e-10) and (std_fm_2 < 1e-10):
            # If std_fm < 1e-10, then zero_flag_fm = 1.
            zero_flag_fm_1 = 1
            zero_flag_fm_2 = 1
        elif (std_fm_1 < 1e-10) and (std_fm_2 >= 1e-10):
            zero_flag_fm_1 = 1
            zero_flag_fm_2 = 0
        elif (std_fm_1 >= 1e-10) and (std_fm_2 < 1e-10):
            zero_flag_fm_1 = 0
            zero_flag_fm_2 = 1
        else:
            zero_flag_fm_1 = 0
            zero_flag_fm_2 = 0

        return zero_flag_fm_1, zero_flag_fm_2

    def pearson_correlation(
            self,
            feature_map1,
            feature_map2
    ):
        """Function to calculate Pearson correlation between two tensors.
      """
        # Concatenate two tensors along a new dimension.
        x = torch.stack([feature_map1, feature_map2])
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

        return corr

    def feature_maps_correlation(
            self,
            model1_name: str,
            model2_name: str,
            model1_layer_name: str,
            model2_layer_name: str,
            model1_feature_map,
            model2_feature_map,
            n_maps1: int,
            n_maps2: int
    ):
        """Function to compute the correlation between feature maps of two layers from different models.

        Parameters:
        -----------
        model1_name : str
            Name of the first model.
        model2_name : str
            Name of the second model.
        model1_layer_name : str
            Name of the layer of the first model to compute the feature maps correlation.
        model2_layer_name : str
            Name of the layer of the second model to compute the feature maps correlation.
        model1_feature_map : torch.Tensor
            3D tensor with shape (batch_size, height, width) containing the feature maps of the chosen layer
            from the first model.
        model2_feature_map : torch.Tensor
            3D tensor with shape (batch_size, height, width) containing the feature maps of the chosen layer
            from the second model.
        n_maps1 : int
            Number of feature maps of the chosen layer from the first model.
        n_maps2 : int
            Number of feature maps of the chosen layer from the second model.

        Returns:
        --------
        pandas.DataFrame
        Dataframe containing the correlation values between each pair of feature maps, along with metadata such
        as the ids and names of the models and layers, and flags indicating if any of the feature maps contain only
        zeros.

        """
        fm_correlation = []

        # Create Object to progress bar
        prog = pyprog.ProgressBar(" ", "", n_maps1)
        # Print Task name
        print(
            f"""Computing feature maps correlation: {model1_name + '_' + model1_layer_name} - {model2_name + '_' + model2_layer_name} \n"""
        )
        # Update Progress Bar
        prog.update()

        for map_idx1 in range(n_maps1):
            # Reshaping it into a one-dimensional tensor
            layer_1_map_1d = model1_feature_map[map_idx1].flatten()

            for map_idx2 in range(n_maps2):
                # Reshaping it into a one-dimensional tensor
                layer_2_map_1d = model2_feature_map[map_idx2].flatten()

                zero_flag_fm_1, zero_flag_fm_2 = self.check_if_map_is_zero(layer_1_map_1d, layer_2_map_1d)

                if (zero_flag_fm_1 == 1) and (zero_flag_fm_2 == 1):
                    corr = 1.0
                elif (zero_flag_fm_1 == 1) and (zero_flag_fm_2 == 0):
                    corr = 0.0
                elif (zero_flag_fm_1 == 0) and (zero_flag_fm_2 == 1):
                    corr = 0.0
                else:
                    zero_flag_fm_1 = 0
                    zero_flag_fm_2 = 0

                    # Calculate Pearson correlation between two feature maps
                    corr = self.pearson_correlation(layer_1_map_1d, layer_2_map_1d)

                # Rename models if they have the same name
                model1_name, model2_name = check_models_name(model1_name, model2_name)

                # Append data to dict
                fm_correlation.append({
                    model1_name + '_fm_id': map_idx1,
                    model2_name + '_fm_id': map_idx2,
                    'correlation': corr,
                    model1_name + '_zero_flag': zero_flag_fm_1,
                    model2_name + '_zero_flag': zero_flag_fm_2,
                    model1_name + '_layer': model1_layer_name,
                    model2_name + '_layer': model2_layer_name
                })

            # Set current status
            prog.set_stat(map_idx1 + 1)
            # Update Progress Bar
            prog.update()

        # Make the Progress Bar final
        prog.end()
        print('\n')

        # Convert the list to a dataframe
        fm_correlation = pd.DataFrame(fm_correlation)
        # Reset indexes
        fm_correlation.reset_index(drop=True, inplace=True)

        return fm_correlation

    def multiple_feature_maps_correlation(
            self,
            layers_metadata1,
            layers_metadata2,
            feature_list_model1,
            feature_list_model2
    ):

        # A dict to store DataFrames of feature maps correlations
        fm_corr_dict = {}
        model1_name = layers_metadata1['model_name']
        model2_name = layers_metadata2['model_name']

        for idx1, layer_name1 in enumerate(layers_metadata1['layer_path']):
            n_maps1 = layers_metadata1['n_maps'][idx1]

            for idx2, layer_name2 in enumerate(layers_metadata2['layer_path']):
                n_maps2 = layers_metadata2['n_maps'][idx2]

                fm_corr_dict[
                    f'{model1_name}_{model2_name}_{layer_name1}_{layer_name2}'] = self.feature_maps_correlation(
                    model1_name=model1_name,
                    model2_name=model2_name,
                    model1_layer_name=layer_name1,
                    model2_layer_name=layer_name2,
                    model1_feature_map=feature_list_model1[:][idx1],
                    model2_feature_map=feature_list_model2[:][idx2],
                    n_maps1=n_maps1,
                    n_maps2=n_maps2
                )

        return fm_corr_dict

    # TODO: documentar a função
    def get_min_correlations(
            self,
            layers_metadata1,
            layers_metadata2,
            fm_corr_dict
    ):
        """Get the min correlation between two differents sets of feature maps.
      """
        fm_corr_min_dict = {}
        for i, key in enumerate(fm_corr_dict.keys()):

            if layers_metadata1['n_maps'][0] >= layers_metadata2['n_maps'][0]:
                column_groupby = fm_corr_dict[key].columns[0]
            else:
                column_groupby = fm_corr_dict[key].columns[1]

            fm_corr_min_dict[f'min_{key}'] = fm_corr_dict[key].loc[
                fm_corr_dict[key].astype(float).groupby(column_groupby)['correlation'].idxmin()].reset_index(drop=True)

        return fm_corr_min_dict

    def get_max_correlations(
            layers_metadata1: dict,
            layers_metadata2: dict,
            feature_maps_corr_dict: dict,
            threshold: float = 0.0,
            same_models: bool = False
    ) -> pd.DataFrame:
        """Obtains the maximum correlation among all combinations of correlations between the feature maps of two models,
        given the metadata of two observed layers, and a dictionary with DataFrames that represent the correlation
        between the feature maps of the two models.

        Parameters:
        -----------
        layers_metadata1: dict
            A dictionary that contains the metadata for the first model's layer of interest.
            Example: {'layer_name': ['Conv1'], 'n_maps': [32]}

        layers_metadata2: dict
            A dictionary that contains the metadata for the second model's layer of interest.
            Example: {'layer_name': ['Conv2'], 'n_maps': [64]}

        feature_maps_corr_dict: dict
            A dictionary that contains DataFrames with the correlation values between the feature maps of the two models.
            Each key of the dictionary should represent a unique pair of layers, and each DataFrame should have the following
            columns: 'model1_fm_id', 'model2_fm_id', 'correlation', 'model1_layer_name', 'model2_layer_name'.

        threshold: float (default 0.0)
            A threshold to filter the feature maps that have a correlation value greater or equal this value.

        same_models: bool (default False)
            A boolean flag that indicates whether the two layers belong to the same model. If True, the function will not
            consider the correlation value of same feature maps from the same layer.

        Returns:
        --------
        max_feature_maps_corr: pd.DataFrame
            A DataFrame that contains the maximum correlation values between all combinations of feature maps of the two
            models. The DataFrame will have the following columns: 'model1_fm_id', 'model2_fm_id', 'correlation',
            'model1_layer_name', 'model2_layer_name'.
        """

        max_feature_maps_corr = []

        for i, key in enumerate(feature_maps_corr_dict.keys()):

            df_aux = pd.DataFrame()

            model1_fm_id_column = feature_maps_corr_dict[key].columns[0]
            model2_fm_id_column = feature_maps_corr_dict[key].columns[1]
            model1_layer_name_column = feature_maps_corr_dict[key].columns[-2]
            model2_layer_name_column = feature_maps_corr_dict[key].columns[-1]

            model1_layer_name = feature_maps_corr_dict[key][model1_layer_name_column][0]
            model2_layer_name = feature_maps_corr_dict[key][model2_layer_name_column][0]

            if layers_metadata1['n_maps'][0] >= layers_metadata2['n_maps'][0]:
                column_to_group = model1_fm_id_column
            else:
                column_to_group = model2_fm_id_column

            # Filter to avoid getting the correlation == 1 of the same feature map
            if same_models:
                df_aux = feature_maps_corr_dict[key].copy()
                df_aux = df_aux[df_aux[model1_fm_id_column] != df_aux[model2_fm_id_column]]
                df_aux = df_aux[df_aux['correlation'] >= threshold]
            else:
                df_aux = feature_maps_corr_dict[key].copy()
                df_aux = df_aux[df_aux['correlation'] >= threshold]

                # Drop the layer name columns
            df_aux.drop([model1_layer_name_column, model2_layer_name_column], axis=1, inplace=True)
            # Get the max correlation
            df_aux = df_aux.loc[
                df_aux
                .astype(float)
                .groupby(column_to_group)['correlation']
                .idxmax()
            ].reset_index(drop=True)

            df_aux[model1_layer_name_column] = model1_layer_name
            df_aux[model2_layer_name_column] = model2_layer_name

            max_feature_maps_corr.append(df_aux)

        max_feature_maps_corr = pd.concat(max_feature_maps_corr)

        return max_feature_maps_corr

    # TODO: documentar a função
    def get_correlation_stats(
            self,
            layers_metadata1,
            layers_metadata2,
            fm_corr_max_or_min_dict
    ):

        """From the maximum or minimum correlation dict, this function calculates the stats of the most frequent
        correlations between feature maps of two models. The stats are: the number of feature maps, their mean,
        median and standard deviation.

        """
        stats_corr = {}

        for i, key in enumerate(fm_corr_max_or_min_dict.keys()):

            if layers_metadata1['n_maps'][0] < layers_metadata2['n_maps'][0]:
                column = fm_corr_max_or_min_dict[key].columns[0]
            else:
                column = fm_corr_max_or_min_dict[key].columns[1]

            aux = fm_corr_max_or_min_dict[key].copy()
            aux[fm_corr_max_or_min_dict[key].columns[0]] = aux[fm_corr_max_or_min_dict[key].columns[0]].astype('str')
            aux[fm_corr_max_or_min_dict[key].columns[1]] = aux[fm_corr_max_or_min_dict[key].columns[1]].astype('str')
            stats_corr[f'stats_{key}'] = pd.DataFrame(aux[column].value_counts())
            # Reset index
            stats_corr[f'stats_{key}'] = stats_corr[f'stats_{key}'].reset_index()
            # Rename columns
            stats_corr[f'stats_{key}'].columns = [column, 'counts']
            # Calculate relative frequency
            stats_corr[f'stats_{key}']['freq%'] = round(
                100 * stats_corr[f'stats_{key}']['counts'] / stats_corr[f'stats_{key}']['counts'].sum(), 4)
            # Calculate mean, median and std
            stats_corr[f'stats_{key}']['mean'] = stats_corr[f'stats_{key}'][column].astype('int64').apply(
                lambda x: fm_corr_max_or_min_dict[key][fm_corr_max_or_min_dict[key][column] == x]['correlation'].mean())
            stats_corr[f'stats_{key}']['median'] = stats_corr[f'stats_{key}'][column].astype('int64').apply(
                lambda x: fm_corr_max_or_min_dict[key][fm_corr_max_or_min_dict[key][column] == x][
                    'correlation'].median())
            stats_corr[f'stats_{key}']['std'] = stats_corr[f'stats_{key}'][column].astype('int64').apply(
                lambda x: fm_corr_max_or_min_dict[key][fm_corr_max_or_min_dict[key][column] == x]['correlation'].std())
            stats_corr[f'stats_{key}']['std'].replace(np.nan, 0, inplace=True)

        return stats_corr

    # TODO: documentar a função
    def correlation_pipeline(
            self,
            img,
            layers_list1,
            layers_list2,
            model1,
            model2,
            save_path=None,
            file_type='csv',
            device='cuda',
            memory_cleaning=False
    ):
        """
      """
        # Initialize ExtractResUNetLayers class
        erl_model1 = ExtractResUNetLayers(model1)
        erl_model2 = ExtractResUNetLayers(model2)

        # Get layers metadata from model
        layers_metadata_model1 = erl_model1.get_layers(layers_list1)
        layers_metadata_model2 = erl_model2.get_layers(layers_list2)

        # Initialize ExtractResUNetMaps
        erm_model1 = ExtractResUNetMaps(model1, dataset=None, image=img, device=device)
        erm_model2 = ExtractResUNetMaps(model2, dataset=None, image=img, device=device)

        # Extract feature maps from layers
        fm_list_model1 = erm_model1.get_multiple_feature_maps(layers_metadata_model1['layer'])
        fm_list_model2 = erm_model2.get_multiple_feature_maps(layers_metadata_model2['layer'])

        # Compute correlation between features maps of two models
        fm_corr_dict = self.multiple_feature_maps_correlation(
            layers_metadata1=layers_metadata_model1,
            layers_metadata2=layers_metadata_model2,
            feature_list_model1=fm_list_model1,
            feature_list_model2=fm_list_model2
        )
        # Compute the maximum correlation between features maps of different layers
        fm_corr_max_dict = self.get_max_correlations(layers_metadata_model1, layers_metadata_model2, fm_corr_dict)
        # Compute the minimum correlation between features maps of different layers
        fm_corr_min_dict = self.get_min_correlations(layers_metadata_model1, layers_metadata_model2, fm_corr_dict)
        # Compute statistics for the most frequently correlated features maps
        stats_max_corr = self.get_correlation_stats(layers_metadata_model1, layers_metadata_model2, fm_corr_max_dict)
        stats_min_corr = self.get_correlation_stats(layers_metadata_model1, layers_metadata_model2, fm_corr_min_dict)

        # TODO Verificar se o diretório é válido.
        if save_path is not None:
            # Save stats_max_corr's dataframes as json or CSV
            if file_type == 'json':
                for (key1, key2, key3, key4, key5) in zip(fm_corr_dict.keys(), fm_corr_max_dict.keys(),
                                                          fm_corr_min_dict.keys(), stats_max_corr.keys(),
                                                          stats_min_corr.keys()):
                    # Export fm_corr_dict
                    fm_corr_dict[key1].to_json(path_or_buf=f'{save_path}/{key1}.json', orient="index")
                    # Export fm_corr_max_dict and fm_corr_min_dict
                    fm_corr_max_dict[key2].to_json(path_or_buf=f'{save_path}/{key2}.json', orient="index")
                    fm_corr_min_dict[key3].to_json(path_or_buf=f'{save_path}/{key3}.json', orient="index")
                    # Export stats_max_corr and stats_min_corr
                    stats_max_corr[key4].to_json(path_or_buf=f'{save_path}/{key4}.json', orient="index")
                    stats_min_corr[key5].to_json(path_or_buf=f'{save_path}/{key5}.json', orient="index")
            elif file_type == 'csv':
                for (key1, key2, key3, key4, key5) in zip(fm_corr_dict.keys(), fm_corr_max_dict.keys(),
                                                          fm_corr_min_dict.keys(), stats_max_corr.keys(),
                                                          stats_min_corr.keys()):
                    # Export fm_corr_dict
                    fm_corr_dict[key1].to_csv(path_or_buf=f'{save_path}/{key1}.csv', sep=',', index=False)
                    # Export fm_corr_max_dict and fm_corr_min_dict
                    fm_corr_max_dict[key2].to_csv(path_or_buf=f'{save_path}/{key2}.csv', sep=',', index=False)
                    fm_corr_min_dict[key3].to_csv(path_or_buf=f'{save_path}/{key3}.csv', sep=',', index=False)
                    # Export stats_max_corr and stats_min_corr
                    stats_max_corr[key4].to_csv(path_or_buf=f'{save_path}/{key4}.csv', sep=',', index=False)
                    stats_min_corr[key5].to_csv(path_or_buf=f'{save_path}/{key5}.csv', sep=',', index=False)
                    # Memory cleaning
        if memory_cleaning:
            del fm_list_model1, fm_list_model2
            gc.collect()
            torch.cuda.empty_cache()
        else:
            return fm_corr_dict, fm_corr_max_dict, fm_corr_min_dict, stats_max_corr, stats_min_corr
