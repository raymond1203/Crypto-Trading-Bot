#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { CommonStack } from "../lib/common-stack";
import { DataStack } from "../lib/data-stack";
import { EcrStack } from "../lib/ecr-stack";
import { TradingStack } from "../lib/trading-stack";

const app = new cdk.App();

const envConfig = app.node.tryGetContext("environments") ?? {};
const targetEnv = app.node.tryGetContext("env") ?? "dev";
const config = envConfig[targetEnv] ?? { envName: "dev", region: "ap-northeast-2" };

const envName: string = config.envName;

const awsEnv: cdk.Environment = {
  account: config.account || process.env.CDK_DEFAULT_ACCOUNT,
  region: config.region || process.env.CDK_DEFAULT_REGION,
};

const common = new CommonStack(app, `CryptoSentinel-Common-${envName}`, {
  envName,
  env: awsEnv,
  description: `CryptoSentinel 공통 리소스 (${envName})`,
});

new EcrStack(app, `CryptoSentinel-ECR-${envName}`, {
  envName,
  env: awsEnv,
  description: `CryptoSentinel ECR 리포지토리 (${envName})`,
});

new DataStack(app, `CryptoSentinel-Data-${envName}`, {
  envName,
  dataBucket: common.dataBucket,
  tradingSecret: common.tradingSecret,
  env: awsEnv,
  description: `CryptoSentinel 데이터 수집 파이프라인 (${envName})`,
});

new TradingStack(app, `CryptoSentinel-Trading-${envName}`, {
  envName,
  env: awsEnv,
  description: `CryptoSentinel 트레이딩 엔진 (${envName})`,
});
