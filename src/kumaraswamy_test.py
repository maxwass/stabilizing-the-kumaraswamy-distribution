import torch

def test_kumaraswamy_shape(self):
    concentration1 = torch.randn(2, 3).abs().requires_grad_()
    concentration0 = torch.randn(2, 3).abs().requires_grad_()
    concentration1_1d = torch.randn(1).abs().requires_grad_()
    concentration0_1d = torch.randn(1).abs().requires_grad_()
    self.assertEqual(
        Kumaraswamy(concentration1, concentration0).sample().size(), (2, 3)
    )
    self.assertEqual(
        Kumaraswamy(concentration1, concentration0).sample((5,)).size(), (5, 2, 3)
    )
    self.assertEqual(
        Kumaraswamy(concentration1_1d, concentration0_1d).sample().size(), (1,)
    )
    self.assertEqual(
        Kumaraswamy(concentration1_1d, concentration0_1d).sample((1,)).size(),
        (1, 1),
    )
    self.assertEqual(Kumaraswamy(1.0, 1.0).sample().size(), ())
    self.assertEqual(Kumaraswamy(1.0, 1.0).sample((1,)).size(), (1,))

# Kumaraswamy distribution is not implemented in SciPy
# Hence these tests are explicit
def test_kumaraswamy_mean_variance(self):
    c1_1 = torch.randn(2, 3).abs().requires_grad_()
    c0_1 = torch.randn(2, 3).abs().requires_grad_()
    c1_2 = torch.randn(4).abs().requires_grad_()
    c0_2 = torch.randn(4).abs().requires_grad_()
    cases = [(c1_1, c0_1), (c1_2, c0_2)]
    for i, (a, b) in enumerate(cases):
        m = Kumaraswamy(a, b)
        samples = m.sample((60000,))
        expected = samples.mean(0)
        actual = m.mean
        error = (expected - actual).abs()
        max_error = max(error[error == error])
        self.assertLess(
            max_error,
            0.01,
            f"Kumaraswamy example {i + 1}/{len(cases)}, incorrect .mean",
        )
        expected = samples.var(0)
        actual = m.variance
        error = (expected - actual).abs()
        max_error = max(error[error == error])
        self.assertLess(
            max_error,
            0.01,
            f"Kumaraswamy example {i + 1}/{len(cases)}, incorrect .variance",
        )

def test_valid_parameter_broadcasting(self):
        # Test correct broadcasting of parameter sizes for distributions that have multiple
        # parameters.
        # example type (distribution instance, expected sample shape)
        valid_examples = [
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]), concentration0=1.0
                ),
                (2,),
            ),
            (
                Kumaraswamy(concentration1=1, concentration0=torch.tensor([1.0, 1.0])),
                (2,),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([1.0]),
                ),
                (2,),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([[1.0], [1.0]]),
                ),
                (2, 2),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([[1.0]]),
                ),
                (1, 2),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0]),
                    concentration0=torch.tensor([[1.0]]),
                ),
                (1, 1),
            ),
        ]

        for dist, expected_size in valid_examples:
            actual_size = dist.sample().size()
            self.assertEqual(
                actual_size,
                expected_size,
                msg=f"{dist} actual size: {actual_size} != expected size: {expected_size}",
            )

            sample_shape = torch.Size((2,))
            expected_size = sample_shape + expected_size
            actual_size = dist.sample(sample_shape).size()
            self.assertEqual(
                actual_size,
                expected_size,
                msg=f"{dist} actual size: {actual_size} != expected size: {expected_size}",
            )

def test_invalid_parameter_broadcasting(self):
    # invalid broadcasting cases; should throw error
    # example type (distribution class, distribution params)
    invalid_examples = [
         (
              Normal,
              {"loc": torch.tensor([[0, 0]]), "scale": torch.tensor([1, 1, 1, 1])},
        ),
        (
            Kumaraswamy,
            {
                "concentration1": torch.tensor([[1, 1]]),
                "concentration0": torch.tensor([1, 1, 1, 1]),
            },
        ),
        (
            Kumaraswamy,
            {
                "concentration1": torch.tensor([[[1, 1, 1], [1, 1, 1]]]),
                "concentration0": torch.tensor([1, 1]),
            },
        )
    ]

    for dist, kwargs in invalid_examples:
        self.assertRaises(RuntimeError, dist, **kwargs)

def test_kumaraswamy_shape_scalar_params(self):
    kumaraswamy = Kumaraswamy(1, 1)
    self.assertEqual(kumaraswamy._batch_shape, torch.Size())
    self.assertEqual(kumaraswamy._event_shape, torch.Size())
    self.assertEqual(kumaraswamy.sample().size(), torch.Size())
    self.assertEqual(kumaraswamy.sample((3, 2)).size(), torch.Size((3, 2)))
    self.assertEqual(
        kumaraswamy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
    )
    self.assertEqual(
        kumaraswamy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
    )
