#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#define TRAIN_SAMPLES 4

void main()
{
	// AND pattern

	// input: X
	double X[TRAIN_SAMPLES][2] =
	{
		0, 0, // class 0
		0, 1, // class 0
		1, 0, // class 0
		1, 1 // class 1
	};
	// target Y
	double Y[TRAIN_SAMPLES] =
	{
		0, 0, 0, 1
	};

	// weight
	double W[3];
	
	//-------------------------------------------
	// 1) Training
	//-------------------------------------------
	
	// initialize W
	for (int i = 0; i < 3; i++)
		W[i] = ((double)rand() / RAND_MAX)*0.5 - 0.25;
				
	unsigned int epoch = 0;
	unsigned int MAX_EPOCH = 500;
	double Etha = 0.05; // learning rate

	double yhat;
	double x1, x2;
	double target;

	while (epoch++ < MAX_EPOCH)
	{
		// 1 epoch (for all training samples)
		// compute deltaWi for each Wi
		//
		double deltaW[3];
		
		deltaW[0] = 0.0;
		deltaW[1] = 0.0;
		deltaW[2] = 0.0;
		for (int i = 0; i < TRAIN_SAMPLES; i++)
		{
			x1 = X[i][0];
			x2 = X[i][0];
			target = Y[i];
			yhat = 1 * W[0] + x1 * W[1] + x2 * W[2];
			
			deltaW[0] += (target - yhat) * 1;
			deltaW[1] += (target - yhat) * x1;
			deltaW[2] += (target - yhat) * x2;
		}
		//
		// update W
		//
		W[0] = W[0] + Etha * (deltaW[0] / TRAIN_SAMPLES);
		W[1] = W[1] + Etha * (deltaW[1] / TRAIN_SAMPLES);
		W[2] = W[2] + Etha * (deltaW[2] / TRAIN_SAMPLES);

		// compute the loss
		
		double loss = 0.0;
		// cost = 0.0;

		for (int i = 0; i < TRAIN_SAMPLES; i++)
		{
			x1 = X[i][0];
			x2 = X[i][1];
			target = Y[i];

			yhat = 1 * W[0] + x1 * W[1] + x2 * W[2];
			loss += (target - yhat) * (target - yhat);
		}
		loss = 0.5 * loss / TRAIN_SAMPLES;
		
		printf("%05d: loss = %10.91f \n", epoch, loss);
	}
	printf("training done\n\n");

	//-------------------------------------------
	// 2) Testing for the training set
	//-------------------------------------------

	for (int i = 0; i < TRAIN_SAMPLES; i++)
	{
		x1 = X[i][0];
		x2 = X[i][1];
		target = Y[i];
		yhat = 1 * W[0] + x1 * W[1] + x2 * W[2];
		printf("%2.1lf %2.1lf (%d) %2.1lf \n", x1, x2, (int)target, yhat);
	}
	printf("training test done\n\n");
	
	//-------------------------------------------
	// 3) Testing for an unknown data
	//-------------------------------------------
	
	x1 = 0.8;
	x2 = 0.7;

	double Threshold = 0.5;
	yhat = 1 * W[0] + x1 * W[1] + x2 * W[2];
	int output_class = (yhat > Threshold) ? 1 : 0;
	printf("%2.1lf %2.1lf (%d) %2.1lf \n", x1, x2, output_class, yhat);
}
