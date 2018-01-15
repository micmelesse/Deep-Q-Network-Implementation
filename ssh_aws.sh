dns=$(cat aws.params)
ssh -i "cos429_aws_key.pem" ubuntu@$dns
