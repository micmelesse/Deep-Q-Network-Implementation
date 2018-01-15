dns=$(cat aws.params)
scp -ri "cos429_aws_key.pem" ubunut@$dns:cos429f17_final_project/model* ./aws_models