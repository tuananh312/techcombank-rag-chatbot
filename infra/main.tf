terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "techcombank-rag-chatbot-tfstate-801651111983"
    key    = "techcombank-rag-chatbot/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "techcombank-rag-chatbot"
}

variable "image_tag" {
  description = "Docker image tag to deploy (set by CI to the git SHA)"
  type        = string
}

# ---------------------------------------------------------------------------
# ECR repository the CI pipeline pushes images to
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# ---------------------------------------------------------------------------
# IAM role for the Lambda function
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "bedrock_invoke" {
  name = "${var.project_name}-bedrock-invoke"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel"]
      Resource = "*"
    }]
  })
}

# ---------------------------------------------------------------------------
# Lambda function (container image) + public Function URL
# ---------------------------------------------------------------------------
resource "aws_lambda_function" "app" {
  function_name = var.project_name
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  timeout       = 60
  memory_size   = 2048

  environment {
    variables = {
      AWS_REGION         = var.aws_region
      EMBED_MODEL_NAME   = "all-MiniLM-L6-v2"
      GENERATION_PROVIDER = "bedrock"
      GEN_MODEL_ID       = "global.anthropic.claude-sonnet-4-6"
    }
  }
}

resource "aws_lambda_function_url" "app_url" {
  function_name      = aws_lambda_function.app.function_name
  authorization_type = "NONE" # demo only — see README for tightening this
}

output "api_url" {
  value = aws_lambda_function_url.app_url.function_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}
