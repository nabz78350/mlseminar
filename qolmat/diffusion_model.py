import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from typing_extensions import Self
from functools import partial
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import pandas as pd
import sklearn as skl
from sklearn import utils as sku
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.impute._base import _BaseImputer
from statsmodels.tsa import seasonal as tsa_seasonal
from qolmat.exceptions import NotDataFrame
from qolmat import metrics


class _Imputer(_BaseImputer):
    """
    Base class for all imputers.
print
    
    Parameters
    ----------
    columnwise : bool, optional
        If True, the imputer will be computed for each column, else it will be computed on the
        whole dataframe, by default False
    shrink : bool, optional
        Indicates if the elementwise imputation method returns a single value, by default False
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None
    imputer_params: Tuple[str, ...]
        List of parameters of the imputer, which can be specified globally or columnwise
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    """

    def __init__(
        self,
        columnwise: bool = False,
        shrink: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        imputer_params: Tuple[str, ...] = (),
        groups: Tuple[str, ...] = (),
    ):
        self.columnwise = columnwise
        self.shrink = shrink
        self.random_state = random_state
        self.imputer_params = imputer_params
        self.groups = groups
        self.missing_values = np.nan

    def get_hyperparams(self, col: Optional[str] = None):
        """
        Filter hyperparameters based on the specified column, the dictionary keys in the form
        name_params/column are only relevent for the specified column and are filtered accordingly.

        Parameters
        ----------
        col : str
            The column name to filter hyperparameters.

        Returns
        -------
        dict
            A dictionary containing filtered hyperparameters.

        """
        hyperparams = {}
        for key in self.imputer_params:
            value = getattr(self, key)
            if "/" not in key:
                name_param = key
                if name_param not in hyperparams:
                    hyperparams[name_param] = value
            elif col is not None:
                name_param, col2 = key.split("/")
                if col2 == col:
                    hyperparams[name_param] = value
        return hyperparams

    def _check_input(self, X: NDArray) -> pd.DataFrame:
        """
        Checks that the input X can be converted into a DataFrame, and returns the corresponding
        dataframe.

        Parameters
        ----------
        X : NDArray
            Array-like to process

        Returns
        -------
        pd.DataFrame
            Formatted dataframe, if the input had no column names then the dataframe columns are
            integers
        """
        if not isinstance(X, (pd.DataFrame)):
            X_np = np.array(X)
            if len(X_np.shape) == 1:
                X_np = X_np.reshape(-1, 1)
            df = pd.DataFrame(X_np, columns=[i for i in range(X_np.shape[1])])
        else:
            df = X
        df = df.astype(float)
        return df

    def _check_dataframe(self, X: NDArray):
        """
        Checks that the input X is a dataframe, otherwise raises an error.

        Parameters
        ----------
        X : NDArray
            Array-like to process

        Raises
        ------
        ValueError
            Input has to be a pandas.DataFrame.
        """
        if not isinstance(X, (pd.DataFrame)):
            raise NotDataFrame(type(X))

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        """
        Fit the imputer on X.

        Parameters
        ----------
        X : pd.DataFrame
            Data matrix on which the Imputer must be fitted.

        Returns
        -------
        self : Self
            Returns self.
        """
        _ = self._validate_data(X, force_all_finite="allow-nan")
        df = self._check_input(X)
        for column in df:
            if df[column].isnull().all():
                raise ValueError("Input contains a column full of NaN")

        self.columns_ = tuple(df.columns)
        self._rng = sku.check_random_state(self.random_state)
        if hasattr(self, "estimator") and hasattr(self.estimator, "random_state"):
            self.estimator.random_state = self._rng

        if self.groups:
            self.ngroups_ = df.groupby(list(self.groups)).ngroup().rename("_ngroup")
        else:
            self.ngroups_ = pd.Series(0, index=df.index).rename("_ngroup")

        self._setup_fit()
        if self.columnwise:
            for col in df.columns:
                self._fit_allgroups(df[[col]], col=col)
        else:
            self._fit_allgroups(df)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe with same shape as `X`, unchanged values, where all nans are replaced
        by non-nan values. Depending on the imputer parameters, the dataframe can be imputed with
        columnwise and/or groupwise methods.
        Also works for numpy arrays, returning numpy arrays, but the use of pandas dataframe is
        advised.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to impute.

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.
        """

        df = self._check_input(X)
        if tuple(df.columns) != self.columns_:
            raise ValueError(
                """The number of features is different from the counterpart in fit.
                Reshape your data"""
            )

        for column in df:
            if df[column].isnull().all():
                raise ValueError("Input contains a column full of NaN")

        cols_with_nans = df.columns[df.isna().any()]

        if self.columnwise:
            df_imputed = df.copy()
            for col in cols_with_nans:
                df_imputed[col] = self._transform_allgroups(df[[col]], col=col)
        else:
            df_imputed = self._transform_allgroups(df)

        if df_imputed.isna().any().any():
            raise AssertionError("Result of imputation contains NaN!")

        df_imputed = df_imputed.astype(float)
        if isinstance(X, (np.ndarray)):
            df_imputed = df_imputed.to_numpy()

        return df_imputed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Returns a dataframe with same shape as `X`, unchanged values, where all nans are replaced
        by non-nan values.
        Depending on the imputer parameters, the dataframe can be imputed with columnwise and/or
        groupwise methods.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to impute.

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.
        """
        self.fit(X)
        return self.transform(X)

    def _fit_transform_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute `df` by the median of each column if it still contains missing values.
        This can introduce data leakage for forward imputers if unchecked.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with missing values.

        Returns
        -------
        pd.DataFrame
            Dataframe df imputed by the median of each column.
        """
        self._check_dataframe(df)
        return df.fillna(df.median())

    def _fit_allgroups(self, df: pd.DataFrame, col: str = "__all__") -> Self:
        """
        Fits the Imputer either on a column, for a columnwise setting, on or all columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"

        Returns
        -------
        Self
            Returns self.

        Raises
        ------
        ValueError
            Input has to be a pandas.DataFrame.
        """

        self._check_dataframe(df)
        fun_on_col = partial(self._fit_element, col=col)
        if self.groups:
            groupby = df.groupby(self.ngroups_, group_keys=False)
            self._dict_fitting[col] = groupby.apply(fun_on_col).to_dict()
        else:
            self._dict_fitting[col] = {0: fun_on_col(df)}

        return self

    def _setup_fit(self) -> None:
        """
        Setup step of the fit function, before looping over the columns.
        """
        self._dict_fitting: Dict[str, Any] = dict()
        return

    def _apply_groupwise(self, fun: Callable, df: pd.DataFrame, **kwargs) -> Any:
        """
        Applies the function `fun`in a groupwise manner to the dataframe `df`.


        Parameters
        ----------
        fun : Callable
            Function applied groupwise to the dataframe with arguments kwargs
        df : pd.DataFrame
            Dataframe on which the function is applied

        Returns
        -------
        Any
            Depends on the function signature
        """
        self._check_dataframe(df)
        fun_on_col = partial(fun, **kwargs)
        if self.groups:
            groupby = df.groupby(self.ngroups_, group_keys=False)
            if self.shrink:
                return groupby.transform(fun_on_col)
            else:
                return groupby.apply(fun_on_col)
        else:
            return fun_on_col(df)

    def _transform_allgroups(self, df: pd.DataFrame, col: str = "__all__") -> pd.DataFrame:
        """
        Impute `df` by applying the specialized method `transform_element` on each group, if
        groups have been given. If the method leaves nan, `fit_transform_fallback` is called in
        order to return a dataframe without nan.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"

        Returns
        -------
        pd.DataFrame
            Imputed dataframe or column

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        df = df.copy()
        imputation_values = self._apply_groupwise(self._transform_element, df, col=col)

        df = df.fillna(imputation_values)
        # fill na by applying imputation method without groups
        if df.isna().any().any():
            imputation_values = self._fit_transform_fallback(df)
            df = df.fillna(imputation_values)

        return df

    @abstractmethod
    def _fit_element(self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0) -> Any:
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return self.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        return self

    @abstractmethod
    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        return df




class ImputerDiffusion(_Imputer):
    """This class inherits from the class _Imputer.
    It is a wrapper for imputers based on diffusion models.
    """

    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        model: Optional[BaseEstimator] = None,
        epochs: int = 100,
        batch_size: int = 100,
        x_valid: pd.DataFrame = None,
        print_valid: bool = False,
        metrics_valid: Tuple[Callable, ...] = (
            metrics.mean_absolute_error,
            metrics.dist_wasserstein,
        ),
        round: int = 10,
        cols_imputed: Tuple[str, ...] = (),
        index_datetime: str = "",
        freq_str: str = "1D",
        columnwise: bool = False,
    ):
        """This class inherits from the class _Imputer.
        It is a wrapper for imputers based on diffusion models.

        Parameters
        ----------
        groups : Tuple[str, ...], optional
            List of column names to group by, by default ()
        model : Optional[BaseEstimator], optional
            Imputer based on diffusion models (e.g., TabDDPM, TsDDPM),
            by default None
        epochs : int, optional
            Number of epochs, by default 10
        batch_size : int, optional
            Batch size, by default 100
        x_valid : pd.DataFrame, optional
            Dataframe for validation, by default None
        print_valid : bool, optional
            Print model performance for after several epochs, by default False
        metrics_valid : Tuple[Callable, ...], optional
            Set of validation metrics, by default ( metrics.mean_absolute_error,
            metrics.dist_wasserstein )
        round : int, optional
            Number of decimal places to round to, for better displaying model
            performance, by default 10
        cols_imputed : Tuple[str, ...], optional
            Name of columns that need to be imputed, by default ()
        index_datetime : str
            Name of datetime-like index.
            It is for processing time-series data, used in diffusion models e.g., TsDDPM.
        freq_str : str
            Frequency string of DateOffset of Pandas.
            It is for processing time-series data, used in diffusion models e.g., TsDDPM.
        """
        self.columnwise = columnwise
        super().__init__(groups=groups, columnwise=self.columnwise)
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_valid = x_valid
        self.print_valid = print_valid
        self.metrics_valid = metrics_valid
        self.round = round
        self.cols_imputed = cols_imputed
        self.index_datetime = index_datetime
        self.freq_str = freq_str

    def _more_tags(self):
        return {
            "non_deterministic": True,
            "_xfail_checks": {
                "check_estimators_pickle": "Diffusion models can return\
                                  different outputs",
                "check_estimators_overwrite_params": "Diffusion models can\
                                    return different outputs",
            },
        }

    def _fit_element(self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0):
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return fitted model

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hp = self._get_params_fit()
        return self.model.fit(df, **hp)

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending on self.groups
        and self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        df_imputed = self.model.predict(df)
        return df_imputed

    def _get_params_fit(self) -> Dict:
        hyperparams = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "x_valid": self.x_valid,
            "print_valid": self.print_valid,
            "metrics_valid": self.metrics_valid,
            "round": self.round,
            "cols_imputed": self.cols_imputed,
        }
        if self.index_datetime != "":
            hyperparams = {
                **hyperparams,
                **{
                    "index_datetime": self.index_datetime,
                    "freq_str": self.freq_str,
                },
            }

        return hyperparams

    def get_summary_training(self) -> Dict:
        return self.model.summary

    def get_summary_architecture(self) -> Dict:
        return {"number_parameters": self.model.num_params, "epsilon_model": self.model._eps_model}