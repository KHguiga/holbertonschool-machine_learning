def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """
        This method generates the Neural Style Transfer image.
        Args:
        iterations (int): the number of iterations to perform gradient
            descent over.
        step (int): the step at which to print information about the
            training, including the final iteration:
            print: Cost at iteration {i}: {J_total}, content
            {J_content}, style {J_style}
                i is the iteration
                J_total is the total cost
                J_content is the content cost
                J_style is the style cost
        lr (float): the learning rate for gradient descent.
        beta1 (float): the beta1 parameter for gradient descent.
        beta2 (float): the beta2 parameter for gradient descent.
        Returns:
        (generated_image, cost) where:
            generated_image is the generated image.
            cost is the cost of the generated image.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        # Initialize the Adam optimizer
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta1,
                                       beta_2=beta2)
        # Initialize the generated image to be a copy of the content image
        generated_image = tf.Variable(self.content_image)
        # Perform optimization
        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            with tf.GradientTape() as tape:
                grads, J_total, J_content, J_style = self.compute_grads(
                    generated_image)

            # Applying gradients to the generated image
            optimizer.apply_gradients([(grads, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, 0, 1))

            # Print each `step`
            if step is not None and i % step == 0:
                print(f"Cost at iteration {i}: {J_total.numpy()}, \
content {J_content.numpy()}, style {J_style.numpy()}")

            # Save the best image
            if J_total < best_cost:
                best_cost = J_total
                prev_image = generated_image
        # Removes the extra dimension from the image
        best_image = prev_image[0]
        return best_image.numpy(), best_cost.numpy()
