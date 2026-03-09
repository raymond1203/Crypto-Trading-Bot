import * as cdk from "aws-cdk-lib";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";

export interface DataStackProps extends cdk.StackProps {
  envName: string;
  dataBucket: s3.IBucket;
  tradingSecret: secretsmanager.ISecret;
}

export class DataStack extends cdk.Stack {
  public readonly collectorFunction: lambda.Function;

  constructor(scope: Construct, id: string, props: DataStackProps) {
    super(scope, id, props);

    const { envName, dataBucket, tradingSecret } = props;

    // Lambda: 데이터 수집 함수
    this.collectorFunction = new lambda.Function(this, "DataCollector", {
      functionName: `cryptosentinel-data-collector-${envName}`,
      runtime: lambda.Runtime.PYTHON_3_14,
      handler: "src.infra.lambda_handlers.data_collector.handler",
      code: lambda.Code.fromAsset("../", {
        exclude: [
          "infra",
          "node_modules",
          "cdk.out",
          ".venv",
          ".git",
          "data",
          "notebooks",
          "docs",
          "tests",
          "*.md",
        ],
      }),
      memorySize: 512,
      timeout: cdk.Duration.minutes(5),
      environment: {
        S3_DATA_BUCKET: dataBucket.bucketName,
        S3_OHLCV_KEY: "ohlcv/BTC_USDT_1h.parquet",
        SYMBOL: "BTC/USDT",
        TIMEFRAME: "1h",
      },
      description: `CryptoSentinel 데이터 수집 (${envName})`,
    });

    // S3 읽기/쓰기 권한
    dataBucket.grantReadWrite(this.collectorFunction);

    // Secrets Manager 읽기 권한 (Binance API 키)
    tradingSecret.grantRead(this.collectorFunction);

    // EventBridge: 매시 정각 트리거
    new events.Rule(this, "HourlySchedule", {
      ruleName: `cryptosentinel-data-collect-${envName}`,
      schedule: events.Schedule.cron({ minute: "0" }),
      targets: [new targets.LambdaFunction(this.collectorFunction)],
      description: "매시 정각 OHLCV 데이터 수집",
    });

    new cdk.CfnOutput(this, "CollectorFunctionArn", {
      value: this.collectorFunction.functionArn,
      exportName: `${envName}-CollectorFunctionArn`,
    });
  }
}
