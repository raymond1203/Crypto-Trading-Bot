import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export interface EcrStackProps extends cdk.StackProps {
  envName: string;
}

export class EcrStack extends cdk.Stack {
  public readonly inferenceRepo: ecr.Repository;

  constructor(scope: Construct, id: string, props: EcrStackProps) {
    super(scope, id, props);

    const { envName } = props;

    // ECR: 추론 Lambda 컨테이너 이미지
    this.inferenceRepo = new ecr.Repository(this, "InferenceRepo", {
      repositoryName: `cryptosentinel-inference-${envName}`,
      removalPolicy:
        envName === "prod"
          ? cdk.RemovalPolicy.RETAIN
          : cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: envName !== "prod",
      lifecycleRules: [
        {
          maxImageCount: 5,
          description: "최근 5개 이미지만 유지",
        },
      ],
      imageScanOnPush: true,
    });

    new cdk.CfnOutput(this, "InferenceRepoUri", {
      value: this.inferenceRepo.repositoryUri,
      exportName: `${envName}-InferenceRepoUri`,
    });
  }
}
