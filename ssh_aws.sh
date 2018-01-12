dns=$(cat aws.params)
ssh -i "cos_429.pem" ubuntu@$dns
