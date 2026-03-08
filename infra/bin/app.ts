#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { CommonStack } from "../lib/common-stack";

const app = new cdk.App();

const envConfig = app.node.tryGetContext("environments") ?? {};
const targetEnv = app.node.tryGetContext("env") ?? "dev";
const config = envConfig[targetEnv] ?? { envName: "dev", region: "ap-northeast-2" };

const envName: string = config.envName;

const awsEnv: cdk.Environment = {
  account: config.account || process.env.CDK_DEFAULT_ACCOUNT,
  region: config.region || process.env.CDK_DEFAULT_REGION,
};

new CommonStack(app, `CryptoSentinel-Common-${envName}`, {
  envName,
  env: awsEnv,
  description: `CryptoSentinel 공통 리소스 (${envName})`,
});
