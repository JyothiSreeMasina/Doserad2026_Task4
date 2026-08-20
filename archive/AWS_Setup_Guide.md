# AWS Setup Guide for DoseRAD2026

Since you've never used AWS before, this guide will walk you through setting up a cloud machine (an EC2 instance) that is identical to what the challenge uses for evaluation.

> [!NOTE]
> AWS is a pay-as-you-go service. The machine we want (`g5.xlarge`) costs about **$1.00 to $1.50 per hour**. You only pay while the machine is running. When you are done for the day, you **must stop the instance** to avoid being charged!

## Step 1: Create an AWS Account
1. Go to [aws.amazon.com](https://aws.amazon.com/) and click **Create an AWS Account**.
2. Follow the prompts. You will need to provide a credit card (you won't be charged unless you use paid services).
3. Once your account is created, sign in to the **AWS Management Console**.

## Step 2: Request a Quota Increase (Crucial!)
By default, new AWS accounts are not allowed to launch expensive GPU instances. You must ask AWS for permission.
1. In the top search bar of the AWS console, type **Service Quotas** and click it.
2. On the left menu, click **AWS services** -> search for **Amazon Elastic Compute Cloud (Amazon EC2)** and click it.
3. In the search bar on that page, type `Running On-Demand G and VT instances`.
4. Select it, and click **Request quota increase**.
5. Change the quota value to **4** (this represents the number of virtual CPUs, and a `g5.xlarge` has 4 vCPUs).
6. Submit the request. It usually takes 12-24 hours for AWS to approve this. 

## Step 3: Launching the Instance
Once your quota is approved, you can start the machine!
1. In the top search bar, search for **EC2** and click it.
2. Click the orange **Launch instance** button.
3. **Name**: Call it `DoseRAD-Machine`.
4. **Application and OS Images (AMI)**: Search for **Deep Learning OSS Nvidia Driver AMI (Ubuntu)**. This is a special image that comes with PyTorch and Nvidia GPU drivers pre-installed. Select it.
5. **Instance Type**: Search for and select `g5.xlarge`.
6. **Key Pair (login)**: Click **Create new key pair**. Name it `doserad-key`. Keep the defaults (RSA, .pem) and click create. **This will download a file to your computer. Keep it safe, it's the password to your machine!**
7. **Storage**: Increase the default storage to at least **100 GB** so you have room for the software and temporary data buffers.
8. Click **Launch instance**.

## Step 4: Connecting to your Machine
1. Once it says "Running", click on the instance ID.
2. Click the **Connect** button at the top.
3. Select the **EC2 Instance Connect** tab and click Connect. A terminal window will open in your browser! You are now inside your cloud computer.
4. *(Optional but recommended)*: You can set up VS Code on your laptop to connect to this machine via SSH using the `.pem` key you downloaded, allowing you to code locally but run it on the cloud GPU.

> [!CAUTION]
> **Always turn it off!** When you are done working, go to the EC2 console, select your instance, click **Instance state**, and click **Stop instance**. If you don't do this, it will cost you ~$24 a day. Do NOT click "Terminate" unless you want to permanently delete the machine and all its files.
