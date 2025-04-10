import unittest
import numpy as np
from eigenfaces import Eigenfaces
from feature_reduction import FeatureReduction
from logistic_regression import LogisticRegression, hyperparameter_tuning
from pca import PCA
from regression import Regression
from smote import SMOTE
from svd_recommender import SVDRecommender
from utilities.local_tests_folder.ef_test import EF_Test
from utilities.local_tests_folder.feature_reduction_test import FeatureReduction_Test
from utilities.local_tests_folder.lr_test import LogisticRegression_Test
from utilities.local_tests_folder.pca_test import PCA_Test
from utilities.local_tests_folder.regression_test import Regression_Test
from utilities.local_tests_folder.smote_test import SMOTE_Test
from utilities.local_tests_folder.svd_recommender_test import SVDRecommender_Test


def print_success_message(msg):
    print(f'UnitTest passed successfully for "{msg}"!')


class TestEigenfaces(unittest.TestCase):
    """
    Tests for Q1: Eigenfaces
    """

    def test_svd(self):
        """
        Test correct implementation of SVD calculation for eigenfaces
        """
        ef = Eigenfaces()
        test_ef = EF_Test()
        U, S, V = ef.svd(test_ef.faces)
        student_U_sum = np.sum(U)
        student_S_sum = np.sum(S)
        student_V_sum = np.sum(V)
        self.assertEqual(U.shape, test_ef.shape_of_U, 'Shape of U is incorrect'
            )
        self.assertEqual(S.shape, test_ef.shape_of_S, 'Shape of S is incorrect'
            )
        self.assertEqual(V.shape, test_ef.shape_of_V, 'Shape of V is incorrect'
            )
        self.assertTrue(np.allclose(student_U_sum, test_ef.svd_U_sum),
            'U is incorrect')
        self.assertTrue(np.allclose(student_S_sum, test_ef.svd_S_sum),
            'S is incorrect')
        success_msg = 'SVD calculation'
        print_success_message(success_msg)

    def test_compress(self):
        """
        Test correct implementation of image compression for all face images
        """
        ef = Eigenfaces()
        test_ef = EF_Test()
        U, S, V = ef.svd(test_ef.faces)
        Uc, Sc, Vc = ef.compress(U, S, V, 2)
        self.assertEqual(np.allclose(np.sum(Uc), test_ef.Uc_sum), True,
            'U compression is incorrect')
        self.assertEqual(np.allclose(np.sum(Sc), test_ef.Sc_sum), True,
            'S compression is incorrect')
        self.assertEqual(np.allclose(np.sum(Vc), test_ef.Vc_sum), True,
            'V compression is incorrect')
        self.assertEqual(Uc.shape, test_ef.shape_of_Uc,
            'Shape of compressed U is incorrect')
        self.assertEqual(Sc.shape, test_ef.shape_of_Sc,
            'Shape of compressed S is incorrect')
        self.assertEqual(Vc.shape, test_ef.shape_of_Vc,
            'Shape of compressed V is incorrect')
        success_msg = 'Image compression'
        print_success_message(success_msg)

    def test_rebuild_svd(self):
        """
        Test correct implementation of SVD reconstruction for all face images
        """
        ef = Eigenfaces()
        test_ef = EF_Test()
        U, S, V = ef.svd(test_ef.faces)
        Uc, Sc, Vc = ef.compress(U, S, V, 2)
        Xrebuild_g = ef.rebuild_svd(Uc, Sc, Vc)
        self.assertEqual(np.allclose(np.sum(Xrebuild_g), test_ef.
            rebuilt_sum), True, 'Reconstruction is incorrect')
        self.assertEqual(Xrebuild_g.shape, test_ef.shape_of_rebuilt,
            'Reconstructed matrix shape is incorrect.')
        success_msg = 'SVD reconstruction'
        print_success_message(success_msg)

    def test_compute_eigenfaces(self):
        ef = Eigenfaces()
        test_ef = EF_Test()
        eigenfaces = ef.compute_eigenfaces(test_ef.faces, 2)
        self.assertEqual(np.allclose(np.sum(eigenfaces), test_ef.
            eigenfaces_sum), True, 'Eigenfaces are incorrect')
        self.assertEqual(eigenfaces.shape, test_ef.eigenfaces_shape,
            'Eigenfaces matrix shape is incorrect')
        success_msg = 'Eigenfaces'
        print_success_message(success_msg)

    def test_compression_ratio(self):
        """
        Test correct implementation of compression ratio calculation for a single image
        """
        ef = Eigenfaces()
        test_ef = EF_Test()
        cr = ef.compression_ratio(test_ef.single_image, 2)
        self.assertEqual(np.allclose(cr, test_ef.compression_ratio_2), True,
            'Compression ratio is incorrect')
        success_msg = 'Compression ratio'
        print_success_message(success_msg)

    def test_recovered_variance_proportion(self):
        """
        Test correct implementation of recovered variance proportion calculation for all face images
        """
        ef = Eigenfaces()
        test_ef = EF_Test()
        U, S, V = ef.svd(test_ef.faces)
        rvp = ef.recovered_variance_proportion(S, 2)
        self.assertEqual(np.allclose(rvp, test_ef.rvp_2), True,
            'Recovered variance proportion is incorrect')
        success_msg = 'Recovered variance proportion'
        print_success_message(success_msg)


class TestSVDRecommender(unittest.TestCase):
    """
    Tests for Q1: SVD Recommender
    """

    def test_recommender_svd(self):
        """
        Test
        """
        recommender = SVDRecommender()
        test_recommender = SVDRecommender_Test()
        R, _, _ = recommender.create_ratings_matrix(test_recommender.ratings_df
            )
        U_k, V_k = recommender.recommender_svd(R, 10)
        my_slice_U_k, my_slice_V_k = test_recommender.get_slice_UV(U_k, V_k)
        correct_slice_U_k, correct_slice_V_k = (test_recommender.slice_U_k,
            test_recommender.slice_V_k)
        self.assertTrue(np.all(U_k.shape == test_recommender.
            U_k_expected_shape),
            'recommender_svd() function returning incorrect U_k shape')
        self.assertTrue(np.all(V_k.shape == test_recommender.
            V_k_expected_shape),
            'recommender_svd() function returning incorrect V_k shape')
        self.assertEqual(np.allclose(my_slice_U_k, correct_slice_U_k), True,
            'recommender_svd() function returning incorrect U_k')
        self.assertEqual(np.allclose(my_slice_V_k, correct_slice_V_k), True,
            'recommender_svd() function returning incorrect V_k')
        success_msg = 'recommender_svd() function'
        print_success_message(success_msg)

    def test_predict(self):
        """
        Test
        """
        recommender = SVDRecommender()
        recommender.load_movie_data()
        test_recommender = SVDRecommender_Test()
        R, users_index, movies_index = recommender.create_ratings_matrix(
            test_recommender.complete_ratings_df)
        mask = np.isnan(R)
        masked_array = np.ma.masked_array(R, mask)
        r_means = np.array(np.mean(masked_array, axis=0))
        R_filled = masked_array.filled(r_means)
        R_filled = R_filled - r_means
        U_k, V_k = recommender.recommender_svd(R_filled, k=8)
        movie_recommendations = recommender.predict(R, U_k, V_k,
            users_index, movies_index, test_recommender.test_user_id,
            test_recommender.movies_pool)
        print('Top 3 Movies the User would want to watch:')
        for movie in movie_recommendations:
            print(movie)
        print('--------------------------------------------------------------')
        self.assertEqual(len(movie_recommendations) == len(test_recommender
            .predict_expected_outputs), True,
            'predict() function is not returning the correct number of recommendations'
            )
        self.assertEqual((movie_recommendations == test_recommender.
            predict_expected_outputs).all(), True,
            'predict() function is not returning the correct recommendations')
        success_msg = 'predict() function'
        print_success_message(success_msg)


class TestPCA(unittest.TestCase):
    """
    Tests for Q2: PCA
    """

    def test_pca(self):
        """
        Test correct implementation of PCA
        """
        pca = PCA()
        test_pca = PCA_Test()
        pca.fit(test_pca.data)
        U, S, V = pca.U, pca.S, pca.V
        self.assertEqual(np.allclose(U, test_pca.U), True, 'U is incorrect')
        self.assertEqual(np.allclose(S, test_pca.S), True, 'S is incorrect')
        self.assertEqual(np.allclose(V, test_pca.V), True, 'V is incorrect')
        success_msg = 'PCA fit'
        print_success_message(success_msg)

    def test_transform(self):
        """
        Test correct implementation of PCA transform
        """
        pca = PCA()
        test_pca = PCA_Test()
        pca.fit(test_pca.data)
        U, S, V = pca.U, pca.S, pca.V
        X_new = pca.transform(test_pca.data)
        self.assertEqual(np.allclose(X_new, test_pca.X_new), True,
            'Transformed data is incorrect')
        success_msg = 'PCA transform'
        print_success_message(success_msg)

    def test_transform_rv(self):
        """
        Test correct implementation of PCA transform with recovered variance
        """
        pca = PCA()
        test_pca = PCA_Test()
        pca.fit(test_pca.data)
        U, S, V = pca.U, pca.S, pca.V
        X_new_rv = pca.transform_rv(test_pca.data, 0.7)
        self.assertEqual(np.allclose(X_new_rv, test_pca.X_new_rv), True,
            'Transformed data is incorrect')
        success_msg = 'PCA transform with recovered variance'
        print_success_message(success_msg)


class TestRegression(unittest.TestCase):
    """
    Tests for Q3: Regression
    """

    def test_rmse(self):
        """
        Test correct implementation of linear regression rmse
        """
        reg = Regression()
        test_reg = Regression_Test()
        rmse_test = np.allclose(reg.rmse(test_reg.predict, test_reg.y_all),
            test_reg.rmse)
        self.assertTrue(rmse_test, 'RMSE is incorrect')
        success_msg = 'RMSE'
        print_success_message(success_msg)

    def test_construct_polynomial_feats(self):
        """
        Test correct implementation of polynomial feature construction
        """
        reg = Regression()
        test_reg = Regression_Test()
        poly_feat_test = np.allclose(reg.construct_polynomial_feats(
            test_reg.x_all, 2), test_reg.construct_poly)
        self.assertTrue(poly_feat_test, 'Polynomial features are incorrect')
        success_msg = 'Polynomial feature construction'
        print_success_message(success_msg)

    def test_predict(self):
        """
        Test correct implementation of linear regression prediction
        """
        reg = Regression()
        test_reg = Regression_Test()
        predict_test = np.allclose(reg.predict(test_reg.x_all_feat,
            test_reg.true_weight), test_reg.predict)
        self.assertTrue(predict_test, 'Prediction is incorrect')
        success_msg = 'Linear regression prediction'
        print_success_message(success_msg)

    def test_linear_fit_closed(self):
        """
        Test correct implementation of closed form linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_closed_test = np.allclose(reg.linear_fit_closed(test_reg.
            x_all_feat, test_reg.y_all), test_reg.linear_closed, rtol=0.0001)
        self.assertTrue(linear_closed_test, 'Weights are incorrect')
        success_msg = 'Closed form linear regression'
        print_success_message(success_msg)

    def test_linear_fit_GD(self):
        """
        Test correct implementation of gradient descent linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_GD, linear_GD_loss = reg.linear_fit_GD(test_reg.x_all_feat,
            test_reg.y_all)
        lgd_test = np.allclose(linear_GD, test_reg.linear_GD)
        lgd_loss_test = np.allclose(linear_GD_loss, test_reg.linear_GD_loss)
        self.assertTrue(lgd_test, 'Weights are incorrect')
        self.assertTrue(lgd_loss_test, 'Loss is incorrect')
        success_msg = 'Gradient descent linear regression'
        print_success_message(success_msg)

    def test_linear_fit_SGD(self):
        """
        Test correct implementation of stochastic gradient descent linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_SGD, linear_SGD_loss = reg.linear_fit_SGD(test_reg.
            x_all_feat, test_reg.y_all, 1)
        lsgd_test = np.allclose(linear_SGD, test_reg.linear_SGD)
        lsgd_loss_test = np.allclose(linear_SGD_loss, test_reg.linear_SGD_loss)
        self.assertTrue(lsgd_test, 'Weights are incorrect')
        self.assertTrue(lsgd_loss_test, 'Loss is incorrect')
        success_msg = 'Stochastic gradient descent linear regression'
        print_success_message(success_msg)

    def test_ridge_fit_closed(self):
        """
        Test correct implementation of closed form ridge regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_closed_test = np.allclose(reg.ridge_fit_closed(test_reg.
            x_all_feat, test_reg.y_all, 10), test_reg.ridge_closed)
        self.assertTrue(ridge_closed_test, 'Weights are incorrect')
        success_msg = 'Closed form ridge regression'
        print_success_message(success_msg)

    def test_ridge_fit_GD(self):
        """
        Test correct implementation of gradient descent ridge regression
        """
        error_atolerance = 1e-10
        reg = Regression()
        test_reg = Regression_Test()
        ridge_GD, ridge_GD_loss = reg.ridge_fit_GD(test_reg.x_all_feat,
            test_reg.y_all, 50000, 10)
        rgd_test = np.allclose(ridge_GD, test_reg.ridge_GD, atol=
            error_atolerance)
        rgd_loss_test = np.allclose(ridge_GD_loss, test_reg.ridge_GD_loss,
            atol=error_atolerance)
        rsgd_bias_incorrect = np.allclose(ridge_GD, test_reg.
            ridge_GD_bias_incorrect, atol=error_atolerance)
        self.assertFalse(rsgd_bias_incorrect,
            'Weights are incorrect. Make sure that you handle the bias term correctly.'
            )
        self.assertTrue(rgd_test, 'Weights are incorrect')
        self.assertTrue(rgd_loss_test, 'Loss is incorrect')
        success_msg = 'Gradient descent ridge regression'
        print_success_message(success_msg)

    def test_ridge_fit_SGD(self):
        """
        Test correct implementation of stochastic gradient descent ridge regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_SGD, ridge_SGD_loss = reg.ridge_fit_SGD(test_reg.x_all_feat,
            test_reg.y_all, 20, 1)
        rsgd_test = np.allclose(ridge_SGD, test_reg.ridge_SGD)
        rsgd_loss_test = np.allclose(ridge_SGD_loss, test_reg.ridge_SGD_loss)
        rsgd_bias_incorrect = np.allclose(ridge_SGD, test_reg.
            ridge_SGD_bias_incorrect)
        self.assertFalse(rsgd_bias_incorrect,
            'Weights are incorrect. Make sure that you handle the bias term correctly.'
            )
        self.assertTrue(rsgd_test, 'Weights are incorrect')
        self.assertTrue(rsgd_loss_test, 'Loss is incorrect')
        success_msg = 'Stochastic gradient descent ridge regression'
        print_success_message(success_msg)

    def test_ridge_cross_validation(self):
        """
        Test correct implementation of ridge regression cross validation
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_cv_test = np.allclose(reg.ridge_cross_validation(test_reg.
            x_all_feat, test_reg.y_all, 3), test_reg.cross_val)
        self.assertTrue(ridge_cv_test, 'Weights are incorrect')
        success_msg = 'Ridge regression cross validation'
        print_success_message(success_msg)


class TestLogisticRegression(unittest.TestCase):
    """
    Tests for Q4: Logistic Regression
    """

    def test_sigmoid(self):
        """
        Test correct implementation of sigmoid
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.sigmoid(test_lr.s)
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            sigmoid_result_slice), 'sigmoid incorrect')
        success_msg = 'Logistic Regression sigmoid'
        print_success_message(success_msg)

    def test_sigmoid(self):
        """
        Test correct implementation of sigmoid
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.sigmoid(test_lr.s)
        self.assertTrue(result.shape == test_lr.s.shape,
            'sigmoid incorrect: check shape')
        result_slice = result[:4, :4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            sigmoid_result_slice), 'sigmoid incorrect')
        success_msg = 'Logistic Regression sigmoid'
        print_success_message(success_msg)

    def test_bias_augment(self):
        """
        Test correct implementation of bias_augment
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.bias_augment(test_lr.x)
        result_slice_sum = np.sum(result[:4, :4])
        self.assertTrue(np.allclose(result_slice_sum, test_lr.
            bias_augment_slice_sum), 'bias_augment incorrect')
        success_msg = 'Logistic Regression bias_augment'
        print_success_message(success_msg)

    def test_predict_probs(self):
        """
        Test correct implementation of predict_probs
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.predict_probs(test_lr.x_aug, test_lr.theta)
        self.assertTrue(result.ndim == 2,
            'predict_probs incorrect: check shape')
        self.assertTrue(result.shape[0] == test_lr.x_aug.shape[0],
            'predict_probs incorrect: check shape')
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            predict_probs_result_slice), 'predict_probs incorrect')
        success_msg = 'Logistic Regression predict_probs'
        print_success_message(success_msg)

    def test_predict_labels(self):
        """
        Test correct implementation of predict_labels
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.predict_labels(test_lr.h_x, test_lr.threshold)
        self.assertTrue(result.ndim == 2,
            'predict_labels incorrect: check shape')
        self.assertTrue(result.shape[0] == test_lr.h_x.shape[0],
            'predict_labels incorrect: check shape')
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            predict_labels_result_slice), 'predict_labels incorrect')
        success_msg = 'Logistic Regression predict_labels'
        print_success_message(success_msg)

    def test_loss(self):
        """
        Test correct implementation of loss
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.loss(test_lr.y, test_lr.h_x)
        self.assertAlmostEqual(result, test_lr.loss_result, msg=
            'loss incorrect')
        success_msg = 'Logistic Regression loss'
        print_success_message(success_msg)

    def test_gradient(self):
        """
        Test correct implementation of gradient
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.gradient(test_lr.x_aug, test_lr.y, test_lr.h_x)
        self.assertTrue(result.ndim == 2, 'gradient incorrect: check shape')
        self.assertTrue(result.shape[0] == test_lr.x_aug.shape[1],
            'gradient incorrect: check shape')
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            gradient_result_slice), 'gradient incorrect')
        success_msg = 'Logistic Regression gradient'
        print_success_message(success_msg)

    def test_accuracy(self):
        """
        Test correct implementation of accuracy
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.accuracy(test_lr.y, test_lr.y_hat)
        self.assertAlmostEqual(result, test_lr.accuracy_result,
            'accuracy incorrect')
        success_msg = 'Logistic Regression accuracy'
        print_success_message(success_msg)

    def test_evaluate(self):
        """
        Test correct implementation of evaluate
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.evaluate(test_lr.x, test_lr.y, test_lr.theta, test_lr.
            threshold)
        self.assertAlmostEqual(result[0], test_lr.evaluate_result[0], msg=
            'evaluate incorrect')
        self.assertAlmostEqual(result[1], test_lr.evaluate_result[1], msg=
            'evaluate incorrect')
        success_msg = 'Logistic Regression evaluate'
        print_success_message(success_msg)

    def test_fit(self):
        """
        Test correct implementation of fit
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.fit(test_lr.x, test_lr.y, test_lr.x, test_lr.y, test_lr
            .lr, test_lr.epochs, test_lr.threshold)
        self.assertTrue(result.ndim == 2, 'fit incorrect: check shape')
        self.assertTrue(result.shape[0] == test_lr.theta.shape[0],
            'fit incorrect: check shape')
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.fit_result_slice),
            'fit incorrect')
        success_msg = 'Logistic Regression fit'
        print_success_message(success_msg)

    def test_thresholding(self):
        """
        Test correct implementation of hyperparameter tuning
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        thresholds = np.array([0.1, 0.3, 0.7, 0.9])
        theta = lr.fit(test_lr.x, test_lr.y, test_lr.x, test_lr.y, test_lr.
            lr, test_lr.epochs, test_lr.threshold)
        result = hyperparameter_tuning(lr, test_lr.x, test_lr.y, theta,
            thresholds)
        self.assertTrue(result[0] == test_lr.threshold_result[0])
        self.assertAlmostEqual(result[1], test_lr.threshold_result[1], msg=
            'accuracies incorrect')
        success_msg = 'Hyperparameter Tuning'
        print_success_message(success_msg)


class TestFeatureReduction(unittest.TestCase):
    """
    Tests for Q5: Feature Reduction
    """

    def test_forward_selection(self):
        fr = FeatureReduction()
        test_fr = FeatureReduction_Test()
        student_features = fr.forward_selection(test_fr.X, test_fr.y,
            test_fr.significance_level)
        correct_features = test_fr.correct_forward
        val = set(student_features) == set(correct_features)
        error_msg = """Your forward selection function did not yield the correct features.
        Common issues with this function are:
        (1) Not using the given regression function
        (2) Not adding a bias term
        (3) Not using the significance level properly
        (4) The order in which the features are removed is incorrect"""
        self.assertTrue(val, error_msg)
        success_msg = 'Forward Selection'
        print_success_message(success_msg)

    def test_backward_elimination(self):
        fr = FeatureReduction()
        test_fr = FeatureReduction_Test()
        student_features = fr.backward_elimination(test_fr.X, test_fr.y,
            test_fr.significance_level)
        correct_features = test_fr.correct_backward
        val = set(student_features) == set(correct_features)
        error_msg = """Your backward elimination function did not yield the correct features.
        Common issues with this function are:
        (1) Not using the given regression function
        (2) Not adding a bias term
        (3) Not using the significance level properly
        (4) The order in which the features are removed is incorrect"""
        self.assertTrue(val, error_msg)
        success_msg = 'Backward Elimination'
        print_success_message(success_msg)


class TestSMOTE(unittest.TestCase):

    def test_simple_confusion_matrix(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        true, pred = sm_data.simple_classification_vectors
        student_conf = sm.generate_confusion_matrix(y_true=true, y_pred=pred)
        correct_conf = sm_data.simple_conf_matrix
        self.assertTrue(student_conf.shape == correct_conf.shape,
            f'The confusion matrix should be a square matrix (u, u), where u is the number of unique labels present. Expected {correct_conf.shape}, got {student_conf.shape}'
            )
        self.assertTrue(isinstance(student_conf[0, 0], (int, np.integer)),
            f'The entries of a confusion matrix should be integers. Expected int-like, got {type(student_conf[0, 0])}. Try setting the dtype of your numpy array.'
            )
        self.assertFalse(np.array_equal(student_conf, correct_conf.T),
            'You have the axes inverted conceptually. Make sure to put "true" and "predicted" on the left and top respectively.'
            )
        self.assertTrue(np.array_equal(student_conf, correct_conf),
            'Confusion matrix incorrectly calculated.')
        print_success_message('simple confusion matrix')

    def test_complex_confusion_matrix(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        true, pred = sm_data.multiclass_classification_vectors
        student_conf = sm.generate_confusion_matrix(y_true=true, y_pred=pred)
        correct_conf = sm_data.multiclass_conf_matrix
        self.assertFalse(student_conf.shape == (2, 2),
            'In section 6.1, you should design your functions for a multiclass classification result. Your function might only be supporting a binary classification.'
            )
        self.assertTrue(student_conf.shape == correct_conf.shape,
            f'The confusion matrix should be a square matrix (u, u), where u is the number of unique labels present in either vector. Expected (4,4), got {student_conf.shape}.'
            )
        self.assertTrue(isinstance(student_conf[0, 0], (int, np.integer)),
            f'The entries of a confusion matrix should be integers. Expected int-like, got {type(student_conf[0, 0])}. Try setting the dtype of your numpy array.'
            )
        self.assertFalse(np.array_equal(student_conf, correct_conf.T),
            'You have the axes inverted conceptually. Make sure to put "true" and "predicted" on the left and top respectively.'
            )
        self.assertTrue(np.array_equal(student_conf, correct_conf),
            'Confusion matrix incorrectly calculated.')
        print_success_message('multiclass confusion matrix')

    def test_interpolate(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        start, end = sm_data.interpolation_vectors
        coeff = sm_data.inter_coeff
        student_point = np.array(sm.interpolate(start, end, coeff))
        correct_point = np.array(sm_data.correct_point)
        flipped_point = np.array(sm_data.flipped_coeff)
        self.assertTrue(student_point.shape == correct_point.shape,
            f'The interpolated point should belong to the same dimension space as the input points. Expected {correct_point.shape}, got {student_point.shape}.'
            )
        self.assertTrue(isinstance(student_point[0], (float, np.floating)),
            f'The interpolated point should be an array of floats. Expected float-like, got {type(student_point[0])}. Try setting the dtype of your numpy array.'
            )
        self.assertFalse(np.allclose(student_point, flipped_point),
            "You flipped the start and the end, or you did the interpolation calculation backwards. You're close!"
            )
        self.assertTrue(np.allclose(student_point, correct_point),
            f"""Your calculation of the interpolation was incorrect. Expected
{correct_point}
 got
{student_point}"""
            )
        print_success_message('interpolation')

    def test_knn(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        points = sm_data.knn_points
        student_nns = sm.k_nearest_neighbors(points, 3)
        correct_nns = sm_data.three_nn
        self.assertTrue(student_nns.shape == correct_nns.shape,
            f'Each point should map to its k nearest neighbors, in the shape (N, k). Expected {correct_nns.shape}, got {student_nns.shape}.'
            )
        self.assertTrue(isinstance(student_nns[0, 0], (int, np.integer)),
            f'The entries of nearest neighbors array should be integer indices. Expected int-like, got {type(student_nns[0, 0])}. Try setting the dtype of your numpy array.'
            )
        is_correct = True
        for i in range(correct_nns.shape[0]):
            if set(correct_nns[i]) != set(student_nns[i]):
                is_correct = False
                break
        self.assertTrue(is_correct,
            f'One or more of your neighborhoods is incorrect.')
        print_success_message('k nearest neighbors')

    def test_smote(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        X = sm_data.X
        y = sm_data.y
        student_output = sm.smote(X, y, 3, [0, 1])
        self.assertTrue(len(student_output) == 2,
            f'You should return an iterable (e.g., Tuple) containing two items: synthetic X and synthetic y. Expected 2, got {len(student_output)}'
            )
        student_synth_X, student_synth_y = student_output
        self.assertTrue(student_synth_y.shape == (30,),
            f'Since |maj|=40 and |min|=10, you should have generated 30 points. Expected (30,), got {student_synth_y.shape}.'
            )
        self.assertTrue(student_synth_X.shape == (30, 4),
            f'Since |maj|=40 and |min|=10, you should have generated 30 points. Expected (30,4), got {student_synth_X.shape}.'
            )
        self.assertTrue(np.all(student_synth_y == 1),
            'The generated y values should be all the minority label: 1')
        X_minority = X[y == 1]
        X_floor = np.min(X_minority, axis=0)
        X_ceil = np.max(X_minority, axis=0)
        is_correct = True
        for i in range(student_synth_X.shape[0]):
            for j in range(student_synth_X.shape[1]):
                if student_synth_X[i, j] < X_floor[j]:
                    is_correct = False
                    break
                if student_synth_X[i, j] > X_ceil[j]:
                    is_correct = False
                    break
        self.assertTrue(is_correct,
            f'You are generating points outside of the convex hull of the minority class. Either you are incorrectly applying interpolate or interpolate is incorrect.'
            )
        print_success_message('SMOTE')

    def test_roc_auc(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        tpr = sm_data.tpr
        fpr = sm_data.fpr
        roc_auc = sm.compute_roc_auc(tpr, fpr)
        self.assertGreater(roc_auc, sm_data.roc_auc_min, msg=
            f'The value of the AUC must be greater than 0, but got {roc_auc}')
        self.assertLessEqual(roc_auc, sm_data.roc_auc_max, msg=
            f'The value of the AUC must not exceed 1, got {roc_auc}')
        self.assertTrue(np.allclose(roc_auc, sm_data.roc_auc, atol=0.1),
            msg=
            f'ROC AUc mismatch: Expected {sm_data.roc_auc}, but got {roc_auc}. Allowed tolerance is 0.1'
            )
        print_success_message('The implementation of compute_roc_auc')

    def test_tpr_fpr(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        y_true = sm_data.y_true_auc
        y_pred = sm_data.y_pred_auc
        tpr_exp = sm_data.tpr
        fpr_exp = sm_data.fpr
        length = len(y_true)
        m = length + 1
        tpr, fpr, thresholds = sm.compute_tpr_fpr(y_true, y_pred, m)
        self.assertEqual(len(tpr), length + 1, msg=
            f'The length of TPR is not m, please check the dimensions of the TPR.'
            )
        self.assertEqual(len(fpr), length + 1, msg=
            f'The length of FPR is not m, please check the dimensions of the FPR.'
            )
        self.assertEqual(len(thresholds), length + 1, msg=
            f'The length of thresholds is not m, please check the dimensions of the thresholds.'
            )
        self.assertTrue(np.all(fpr[:-1] <= fpr[1:]), msg=
            'FPR is not sorted in the ascending order. Please ensure that it is sorted to calculate the AUC correctly.'
            )
        self.assertTrue(np.allclose(tpr, tpr_exp, atol=0.0001), msg=
            'TPR values do not match expected the expected values.')
        self.assertTrue(np.allclose(fpr, fpr_exp, atol=0.0001), msg=
            f'FPR values do not match expected the expected values.')
        print_success_message('TPR, FPR and threshold')
