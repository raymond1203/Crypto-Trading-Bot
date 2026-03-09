import * as cdk from "aws-cdk-lib";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import { Construct } from "constructs";

export interface TradingStackProps extends cdk.StackProps {
  envName: string;
}

export class TradingStack extends cdk.Stack {
  public readonly positionsTable: dynamodb.Table;
  public readonly tradesTable: dynamodb.Table;
  public readonly botStateTable: dynamodb.Table;

  constructor(scope: Construct, id: string, props: TradingStackProps) {
    super(scope, id, props);

    const { envName } = props;

    // DynamoDB: 현재 오픈 포지션
    this.positionsTable = new dynamodb.Table(this, "PositionsTable", {
      tableName: `cryptosentinel-positions-${envName}`,
      partitionKey: { name: "symbol", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy:
        envName === "prod"
          ? cdk.RemovalPolicy.RETAIN
          : cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: envName === "prod" ? { pointInTimeRecoveryEnabled: true } : undefined,
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
      pointInTimeRecoverySpecification: envName === "prod" ? { pointInTimeRecoveryEnabled: true } : undefined,
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
  }
}
