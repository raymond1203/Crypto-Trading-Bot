import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";

export interface CommonStackProps extends cdk.StackProps {
  envName: string;
}

export class CommonStack extends cdk.Stack {
  public readonly dataBucket: s3.Bucket;
  public readonly modelBucket: s3.Bucket;
  public readonly tradingSecret: secretsmanager.Secret;

  constructor(scope: Construct, id: string, props: CommonStackProps) {
    super(scope, id, props);

    const { envName } = props;

    // S3: 데이터 레이크 버킷
    this.dataBucket = new s3.Bucket(this, "DataBucket", {
      bucketName: `cryptosentinel-data-${envName}`,
      removalPolicy:
        envName === "prod"
          ? cdk.RemovalPolicy.RETAIN
          : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: envName !== "prod",
      intelligentTieringConfigurations: [
        {
          name: "auto-tier",
          archiveAccessTierTime: cdk.Duration.days(90),
          deepArchiveAccessTierTime: cdk.Duration.days(180),
        },
      ],
      lifecycleRules: [
        {
          prefix: "raw/",
          expiration: cdk.Duration.days(365),
        },
      ],
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: false,
    });

    // S3: 모델 아티팩트 버킷
    this.modelBucket = new s3.Bucket(this, "ModelBucket", {
      bucketName: `cryptosentinel-models-${envName}`,
      removalPolicy:
        envName === "prod"
          ? cdk.RemovalPolicy.RETAIN
          : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: envName !== "prod",
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: true,
    });

    // Secrets Manager: Binance API + Telegram 봇 토큰
    this.tradingSecret = new secretsmanager.Secret(this, "TradingSecret", {
      secretName: `cryptosentinel/${envName}/trading-credentials`,
      description: `CryptoSentinel 트레이딩 자격 증명 (${envName})`,
      generateSecretString: {
        secretStringTemplate: JSON.stringify({
          BINANCE_API_KEY: "",
          BINANCE_API_SECRET: "",
          TELEGRAM_BOT_TOKEN: "",
          TELEGRAM_CHAT_ID: "",
        }),
        generateStringKey: "placeholder",
      },
    });

    // CloudFormation Outputs (다른 스택에서 참조)
    new cdk.CfnOutput(this, "DataBucketArn", {
      value: this.dataBucket.bucketArn,
      exportName: `${envName}-DataBucketArn`,
    });

    new cdk.CfnOutput(this, "ModelBucketArn", {
      value: this.modelBucket.bucketArn,
      exportName: `${envName}-ModelBucketArn`,
    });

    new cdk.CfnOutput(this, "TradingSecretArn", {
      value: this.tradingSecret.secretArn,
      exportName: `${envName}-TradingSecretArn`,
    });
  }
}
