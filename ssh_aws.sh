dns=$(cat dns.txt)
ssh -i "cos_429.pem" ubuntu@$dns

