dns=$(cat aws.params)
scp -ri "cos429_aws_key.pem" ubuntu@$dns:cos429_f17_final_project/model* ./aws_model
