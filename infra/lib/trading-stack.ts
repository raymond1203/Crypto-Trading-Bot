import * as cdk from "aws-cdk-lib";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";

export interface TradingStackProps extends cdk.StackProps {
  envName: string;
  dataBucket: s3.IBucket;
  modelBucket: s3.IBucket;
  tradingSecret: secretsmanager.ISecret;
  inferenceRepo: ecr.IRepository;
}

export class TradingStack extends cdk.Stack {
  public readonly positionsTable: dynamodb.Table;
  public readonly tradesTable: dynamodb.Table;
  public readonly botStateTable: dynamodb.Table;
  public readonly inferenceFunction: lambda.Function;

  constructor(scope: Construct, id: string, props: TradingStackProps) {
    super(scope, id, props);

    const { envName, dataBucket, modelBucket, tradingSecret, inferenceRepo } =
      props;

    // DynamoDB: 현재 오픈 포지션
    this.positionsTable = new dynamodb.Table(this, "PositionsTable", {
      tableName: `cryptosentinel-positions-${envName}`,
      partitionKey: { name: "symbol", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy:
        envName === "prod"
          ? cdk.RemovalPolicy.RETAIN
          : cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification:
        envName === "prod"
          ? { pointInTimeRecoveryEnabled: true }
          : undefined,
    });

    // DynamoDB: 완료된 거래 이력
    this.tradesTable = new dynamodb.Table(this, "TradesTable", {
      tableName: `cryptosentinel-trades-${envName}`,
      partitionKey: { name: "symbol", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "entry_time", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy:
        envName === "prod"
          ? cdk.RemovalPolicy.RETAIN
          : cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification:
        envName === "prod"
          ? { pointInTimeRecoveryEnabled: true }
          : undefined,
    });

    // DynamoDB: 봇 제어 상태
    this.botStateTable = new dynamodb.Table(this, "BotStateTable", {
      tableName: `cryptosentinel-bot-state-${envName}`,
      partitionKey: { name: "key", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy:
        envName === "prod"
          ? cdk.RemovalPolicy.RETAIN
          : cdk.RemovalPolicy.DESTROY,
    });

    // Lambda: 추론 함수 (컨테이너 이미지)
    this.inferenceFunction = new lambda.DockerImageFunction(
      this,
      "InferenceFunction",
      {
        functionName: `cryptosentinel-inference-${envName}`,
        code: lambda.DockerImageCode.fromEcr(inferenceRepo),
        memorySize: 1024,
        timeout: cdk.Duration.minutes(5),
        ephemeralStorageSize: cdk.Size.mebibytes(1024),
        environment: {
          S3_DATA_BUCKET: dataBucket.bucketName,
          S3_MODEL_BUCKET: modelBucket.bucketName,
          S3_OHLCV_KEY: "ohlcv/BTC_USDT_1h.parquet",
          S3_MODEL_PREFIX: "models/latest/",
          SYMBOL: "BTC/USDT",
          TABLE_PREFIX: "cryptosentinel",
          ENV_NAME: envName,
        },
        description: `CryptoSentinel 추론 엔진 (${envName})`,
      },
    );

    // S3 읽기 권한
    dataBucket.grantRead(this.inferenceFunction);
    modelBucket.grantRead(this.inferenceFunction);

    // Secrets Manager 읽기 권한 (Binance API 키)
    tradingSecret.grantRead(this.inferenceFunction);

    // DynamoDB 읽기/쓰기 권한
    this.positionsTable.grantReadWriteData(this.inferenceFunction);
    this.tradesTable.grantReadWriteData(this.inferenceFunction);
    this.botStateTable.grantReadWriteData(this.inferenceFunction);

    // EventBridge: 매시 5분 트리거 (데이터 수집 5분 후)
    new events.Rule(this, "InferenceSchedule", {
      ruleName: `cryptosentinel-inference-${envName}`,
      schedule: events.Schedule.cron({ minute: "5" }),
      targets: [new targets.LambdaFunction(this.inferenceFunction)],
      description: "매시 5분 앙상블 추론 실행",
    });

    // Outputs
    new cdk.CfnOutput(this, "PositionsTableName", {
      value: this.positionsTable.tableName,
      exportName: `${envName}-PositionsTableName`,
    });

    new cdk.CfnOutput(this, "TradesTableName", {
      value: this.tradesTable.tableName,
      exportName: `${envName}-TradesTableName`,
    });

    new cdk.CfnOutput(this, "BotStateTableName", {
      value: this.botStateTable.tableName,
      exportName: `${envName}-BotStateTableName`,
    });

    new cdk.CfnOutput(this, "InferenceFunctionArn", {
      value: this.inferenceFunction.functionArn,
      exportName: `${envName}-InferenceFunctionArn`,
    });
  }
}
