## SparkALS_AWS_SageMaker
ALS based recommendation Engine Build on Apache spark &amp; served on AWS Sagemaker
Collaborative Filter Recommendation Engine built on Apache Spark:
1) Tools & libraries used:
              Python 3.6
              Python Flask for serverless architecture & scaling on AWS
             HADOOP_VERSION 2.7
              Apache Spark 2.3.0
              Python Boto3
             Scikit-learn
              Jupyter Notebook (In production integrated into Sagemaker)
             AWS EC2
             AWS Sagemaker
            AWS S3
           AWS ECS : AmazonEC2ContainerRegistry
            Docker

2) Problem to solve: 
Building recommendation engine with retail explicit feedback data (consisting only of customer ID, prod ID, qty purchased and date of purchase).
The solution is to predict which customer would buy what products.

Explicit feedback data (Sales history data) is all that was available to build a recommendation engine. Due to the lack of contextual information or user demographics, we used Collaborative filtering approach to build recommendation engine. 
We used 'Alternating Least Square (ALS)' a variant of 'Non-Negative Matrix Factorization'.

Model builds two matrices from the Sales data called 'user latent factors' and 'item latent factors’. Due to large number of customers & products in the sales history and the data being explicit & non-negative in nature, leads to a very large sparse matrix.
Values in the 'User Latent Matrix’ represent latent users’ preferences & those in 'Product Latent matrix' represent latent product quality/attributes. This method of creating or breaking the sales data into 2 matrices is called matrix decomposition and since the values are Non-negative, the approach is called 'Non-Negative Matrix Factorization'.

The next step is ALS: we alternatively multiple matrices by keeping one matrix in each iteration constant while changing the other matrix values (the gradient by which to change the values depends on : Lambda value & convergence is effected by the the type of regularization used).
The above step (ALS) is repeated as many times as we define the 'iteration' value to be.
For each iteration a difference is calculated between the matrix (derived from matrix multiplication) and the ground truth (from sales history). A complete convergence is not feasible and hence number of iterations define the quality of final result (of course other parameters also influence the results).
The final matrix is values (or we could call it score) is used to rank the top-recommended products for a given customer.

Transformations of input data & results:

But, we cannot use this output as-is, because:
1) Cold-Start problem
           a) New products have minimal purchase history.
           b) New customers have minimal purchase history.
2) Popular products invariably will have highest score across a large population of customers.

To overcome this few custom transformations were implemented:
       a) Include & exclude options provided to the end user:
Let's say, 
> we must exclude the recommendations of popular products.
> We must include few new products in the recommendations, though their actual score is low in the final matrix.
> New customers could be included to force recommendations. 

3) Architecture discussion:
ALS application which is built using Apache Spark is containerized using Docker (Docker image) & AWS Sagemaker is used for training & Serving the algorithm.
S3 is the storage which holds data.
Hyper parameter values are in als_spec.json
Jupyter notebook is part of AWS Sagemker where the code is written.
Directory structure:
Log into Sagemaker and place “alsriaz” from the attached file to the email in opt/ml in Sagemaker.  
/opt/ml
├── input
│   ├── config
│   │   ├── als_spec.json
│   └── data
│       └── Online Retail.csv
├── model
│   └── <model files> (can be seen only after training in Sagemaker)
└── output
└── output.csv

Input : /opt/ml/input/config contains information to control how your program runs. als_spec.json is a JSON-formatted dictionary of hyperparameter names to values. These values will always be strings, so you may need to convert them.
•	
                Below are ALS parameter
{
	"ranking_type":"count"
	,"max_itter": 5
	,"retention_date":"20180205"
	,"retention_days": 10
	,"reg_param" : 0.01
	,"als_rank" : 10
	,"reco_item" : 3
	,"group_size":4099
}
•	/opt/ml/input/data/Online Retail.csv/ (for File mode) contains the input data for that channel. The channels are created based on the call to CreateTrainingJob (is part of Sagemaker training) but it's generally important that channels match what the algorithm expects. The files for each channel will be copied from S3 to this directory, preserving the tree structure indicated by the S3 key structure. 
The output
•	/opt/ml/model/ is the directory where you write the model that your algorithm generates. It can be a single file or a whole directory tree. SageMaker will package any files in this directory into a compressed tar archive file. This file will be available at the S3 location returned in the DescribeTrainingJob result.
•	/opt/ml/output is a directory where the algorithm can write a file failure that describes why the job failed. The contents of this file will be returned in the FailureReason field of the DescribeTrainingJob result. 

Running your container during hosting
Hosting has a very different model that training because hosting is responding to inference requests that come in via HTTP. In this example, we use our recommended Python serving stack to provide robust and scalable serving of inference requests:
This stack is implemented in the sample code here and you can mostly just leave it alone. 
Amazon SageMaker uses two URLs in the container:
•	/ping will receive GET requests from the infrastructure. Your program returns 200 if the container is up and accepting requests.
•	/invocations is the endpoint that receives client inference POST requests. The format of the request and the response is up to the algorithm. If the client supplied ContentType and Accept headers, these will be passed in as well. 
•	The container will have the model files in the same place they were written during training:
/opt/ml
└── model
   └── <model files>

•	The parts of the sample container
•	In the container directory are all the components you need to package the sample algorithm for Amazon SageMager:

├── Dockerfile
├── build_and_push.sh
└── alsriaz
    ├── nginx.conf
    ├── predictor.py
    ├── serve
    ├── train
└── wsgi.py


•	Dockerfile describes how to build your Docker container image. More details below.
•	build_and_push.sh is a script that users the Dockerfile to build your container images and then pushes it to ECR. We'll invoke the commands directly later in this notebook, but you can just copy and run the script for your own algorithms.
•	alsriaz is the directory which contains the files that will be installed in the container.
The files that we'll put in the container are:
•	nginx.conf is the configuration file for the nginx front-end. Generally, you should be able to take this file as-is.
•	predictor.py is the program that actually implements the Flask web server and the decision tree predictions for this app. You'll want to customize the actual prediction parts to your application. Since this algorithm is simple, we do all the processing here in this file, but you may choose to have separate files for implementing your custom logic.
•	serve is the program started when the container is started for hosting. It simply launches the gunicorn server which runs multiple instances of the Flask app defined in predictor.py. You should be able to take this file as-is.
•	train is the program that is invoked when the container is run for training. You will modify this program to implement your training algorithm.
•	wsgi.py is a small wrapper used to invoke the Flask app. You should be able to take this file as-is.
The Dockerfile
The Dockerfile describes the image that we want to build. You can think of it as describing the complete operating system installation of the system that you want to run. A Docker container running is quite a bit lighter than a full operating system, however, because it takes advantage of Linux on the host machine for the basic operations. 
For the Python science stack, we will start from a standard Ubuntu installation and run the normal tools to install the things needed by scikit-learn. Finally, we add the code that implements our specific algorithm to the container and set up the right environment to run under.
Along the way, we clean up extra space. This makes the container smaller and faster to start.
Let's look at the Dockerfile for the example:

!cat container/Dockerfile
Building and registering the container
The following shell code shows how to build the container image using docker build and push the container image to ECR using docker push. 
This code looks for an ECR repository in the account you're using and the current default region (if you're using a SageMaker notebook instance, this will be the region where the notebook instance was created). If the repository doesn't exist, the script will create it.

%%sh

# The name of our algorithm
algorithm_name=decision-trees-sample

cd container

chmod +x alsriaz/train
chmod +x alsriaz /serve

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# On a SageMaker Notebook Instance, the docker daemon may need to be restarted in order
# to detect your network configuration correctly.  (This is a known issue.)
if [ -d "/home/ec2-user/SageMaker" ]; then
  sudo service docker restart
fi

docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}


