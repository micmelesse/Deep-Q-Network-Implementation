dns=$(cat aws.config)
ssh -i "cos_429.pem" ubuntu@$dns

