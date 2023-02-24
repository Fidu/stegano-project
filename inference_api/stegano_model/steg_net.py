# imports
import torch
from torch import nn
import torch.nn.functional as F

SIZE_IMAGE = 64

class StegNet(nn.Module):
    def __init__(self):
        super(StegNet, self).__init__()
        self.define_encoder()
        self.define_decoder()

    def define_encoder(self):
        # layer1
        self.encoder_payload_1 = nn.Conv2d(1, SIZE_IMAGE, kernel_size=3, padding=1)
        self.encoder_source_1 = nn.Conv2d(3, SIZE_IMAGE, kernel_size=3, padding=1)

        # layer2
        self.encoder_payload_2 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )
        self.encoder_source_2 = nn.Conv2d(
            SIZE_IMAGE * 2, SIZE_IMAGE * 2, kernel_size=3, padding=1
        )
        self.encoder_source_21 = nn.Conv2d(
            SIZE_IMAGE * 2, SIZE_IMAGE, kernel_size=3, padding=1
        )

        # layer3
        self.encoder_payload_3 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )
        self.encoder_source_3 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )

        # layer4
        self.encoder_payload_4 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )
        self.encoder_source_4 = nn.Conv2d(
            SIZE_IMAGE * 4, SIZE_IMAGE * 2, kernel_size=3, padding=1
        )
        self.encoder_source_41 = nn.Conv2d(
            SIZE_IMAGE * 2, SIZE_IMAGE, kernel_size=3, padding=1
        )

        # layer5
        self.encoder_payload_5 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )
        self.encoder_source_5 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )

        # layer6
        self.encoder_payload_6 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )
        self.encoder_source_6 = nn.Conv2d(
            SIZE_IMAGE * 6, SIZE_IMAGE * 4, kernel_size=3, padding=1
        )
        self.encoder_source_61 = nn.Conv2d(
            SIZE_IMAGE * 4, SIZE_IMAGE * 2, kernel_size=3, padding=1
        )
        self.encoder_source_62 = nn.Conv2d(
            SIZE_IMAGE * 2, SIZE_IMAGE, kernel_size=3, padding=1
        )

        # layer7
        self.encoder_payload_7 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )
        self.encoder_source_7 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )

        # layer8
        self.encoder_payload_8 = nn.Conv2d(
            SIZE_IMAGE, SIZE_IMAGE, kernel_size=3, padding=1
        )
        self.encoder_source_8 = nn.Conv2d(
            SIZE_IMAGE * 8, SIZE_IMAGE * 4, kernel_size=3, padding=1
        )
        self.encoder_source_81 = nn.Conv2d(
            SIZE_IMAGE * 4, SIZE_IMAGE * 2, kernel_size=3, padding=1
        )
        self.encoder_source_82 = nn.Conv2d(
            SIZE_IMAGE * 2, SIZE_IMAGE, kernel_size=3, padding=1
        )

        # layer9
        self.encoder_source_9 = nn.Conv2d(
            SIZE_IMAGE, int(SIZE_IMAGE / 2), kernel_size=1
        )

        # layer10
        self.encoder_source_10 = nn.Conv2d(
            int(SIZE_IMAGE / 2), int(SIZE_IMAGE / 4), kernel_size=1
        )

        # layer11
        self.encoder_source_11 = nn.Conv2d(int(SIZE_IMAGE / 4), 3, kernel_size=1)

    def define_decoder(self):
        # layer1
        self.decoder_layers1 = nn.Conv2d(3, SIZE_IMAGE * 8, kernel_size=3, padding=1)

        # layer2
        self.decoder_layers2 = nn.Conv2d(
            SIZE_IMAGE * 8, SIZE_IMAGE * 4, kernel_size=3, padding=1
        )

        # layer3
        self.decoder_layers3 = nn.Conv2d(
            SIZE_IMAGE * 4, SIZE_IMAGE * 2, kernel_size=3, padding=1
        )

        # layer4
        self.decoder_layers4 = nn.Conv2d(
            SIZE_IMAGE * 2, SIZE_IMAGE * 2, kernel_size=3, padding=1
        )

        # layer5
        self.decoder_layers5 = nn.Conv2d(
            SIZE_IMAGE * 2, SIZE_IMAGE, kernel_size=3, padding=1
        )

        # payload_decoder
        self.decoder_payload1 = nn.Conv2d(
            SIZE_IMAGE, int(SIZE_IMAGE / 2), kernel_size=3, padding=1
        )
        self.decoder_payload2 = nn.Conv2d(
            int(SIZE_IMAGE / 2), int(SIZE_IMAGE / 2), kernel_size=3, padding=1
        )

        self.decoder_payload3 = nn.Conv2d(
            int(SIZE_IMAGE / 2), int(SIZE_IMAGE / 4), kernel_size=3, padding=1
        )
        self.decoder_payload4 = nn.Conv2d(
            int(SIZE_IMAGE / 4), int(SIZE_IMAGE / 4), kernel_size=3, padding=1
        )

        self.decoder_payload5 = nn.Conv2d(
            int(SIZE_IMAGE / 4), 3, kernel_size=3, padding=1
        )
        self.decoder_payload6 = nn.Conv2d(3, 1, kernel_size=3, padding=1)

        # source_decoder
        self.decoder_source1 = nn.Conv2d(
            SIZE_IMAGE, int(SIZE_IMAGE / 2), kernel_size=3, padding=1
        )
        self.decoder_source2 = nn.Conv2d(
            int(SIZE_IMAGE / 2), int(SIZE_IMAGE / 2), kernel_size=3, padding=1
        )

        self.decoder_source3 = nn.Conv2d(
            int(SIZE_IMAGE / 2), int(SIZE_IMAGE / 4), kernel_size=3, padding=1
        )
        self.decoder_source4 = nn.Conv2d(
            int(SIZE_IMAGE / 4), int(SIZE_IMAGE / 4), kernel_size=3, padding=1
        )

        self.decoder_source5 = nn.Conv2d(
            int(SIZE_IMAGE / 4), 3, kernel_size=3, padding=1
        )
        self.decoder_source6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        source, payload = x

        s = source.view((-1, 3, SIZE_IMAGE, SIZE_IMAGE))
        p = payload.view((-1, 1, SIZE_IMAGE, SIZE_IMAGE))

        # --------------------------- Encoder -------------------------
        # layer1
        p = F.relu(self.encoder_payload_1(p))
        s = F.relu(self.encoder_source_1(s))

        # layer2
        p = F.relu(self.encoder_payload_2(p))
        s1 = torch.cat((s, p), 1)
        s = F.relu(self.encoder_source_2(s1))
        s = F.relu(self.encoder_source_21(s1))

        # layer3
        p = F.relu(self.encoder_payload_3(p))
        s = F.relu(self.encoder_source_3(s))

        # layer4
        p = F.relu(self.encoder_payload_4(p))
        s2 = torch.cat((s, p, s1), 1)
        s = F.relu(self.encoder_source_4(s2))
        s = F.relu(self.encoder_source_41(s))

        # layer5
        p = F.relu(self.encoder_payload_5(p))
        s = F.relu(self.encoder_source_5(s))

        # layer6
        p = F.relu(self.encoder_payload_6(p))
        s3 = torch.cat((s, p, s2), 1)
        s = F.relu(self.encoder_source_6(s3))
        s = F.relu(self.encoder_source_61(s))
        s = F.relu(self.encoder_source_62(s))

        # layer7
        p = F.relu(self.encoder_payload_7(p))
        s = F.relu(self.encoder_source_7(s))

        # layer8
        p = F.relu(self.encoder_payload_8(p))
        s4 = torch.cat((s, p, s3), 1)
        s = F.relu(self.encoder_source_8(s4))
        s = F.relu(self.encoder_source_81(s))
        s = F.relu(self.encoder_source_82(s))

        # layer9
        s = F.relu(self.encoder_source_9(s))

        # layer10
        s = F.relu(self.encoder_source_10(s))

        # layer11
        encoder_output = self.encoder_source_11(s)

        # -------------------- Decoder --------------------------

        d = encoder_output.view(-1, 3, SIZE_IMAGE, SIZE_IMAGE)

        # layer1
        d = F.relu(self.decoder_layers1(d))
        d = F.relu(self.decoder_layers2(d))

        # layer3
        d = F.relu(self.decoder_layers3(d))
        d = F.relu(self.decoder_layers4(d))

        init_d = F.relu(self.decoder_layers5(d))

        # ---------------- decoder_payload ----------------

        # layer 1 & 2
        d = F.relu(self.decoder_payload1(init_d))
        d = F.relu(self.decoder_payload2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_payload3(d))
        d = F.relu(self.decoder_payload4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_payload5(d))
        decoded_payload = self.decoder_payload6(d)

        # ---------------- decoder_source ----------------

        # layer 1 & 2
        d = F.relu(self.decoder_source1(init_d))
        d = F.relu(self.decoder_source2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_source3(d))
        d = F.relu(self.decoder_source4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_source5(d))
        decoded_source = self.decoder_source6(d)

        return encoder_output, decoded_payload, decoded_source

    # TODO esto es el ejercicio 3
    def predict_encoder(self, source_image, payload_image):
        s = source_image.view((-1, 3, SIZE_IMAGE, SIZE_IMAGE))
        p = payload_image.view((-1, 1, SIZE_IMAGE, SIZE_IMAGE))

        # --------------------------- Encoder -------------------------
        # layer1
        p = F.relu(self.encoder_payload_1(p))
        s = F.relu(self.encoder_source_1(s))

        # layer2
        p = F.relu(self.encoder_payload_2(p))
        s1 = torch.cat((s, p), 1)
        s = F.relu(self.encoder_source_2(s1))
        s = F.relu(self.encoder_source_21(s1))

        # layer3
        p = F.relu(self.encoder_payload_3(p))
        s = F.relu(self.encoder_source_3(s))

        # layer4
        p = F.relu(self.encoder_payload_4(p))
        s2 = torch.cat((s, p, s1), 1)
        s = F.relu(self.encoder_source_4(s2))
        s = F.relu(self.encoder_source_41(s))

        # layer5
        p = F.relu(self.encoder_payload_5(p))
        s = F.relu(self.encoder_source_5(s))

        # layer6
        p = F.relu(self.encoder_payload_6(p))
        s3 = torch.cat((s, p, s2), 1)
        s = F.relu(self.encoder_source_6(s3))
        s = F.relu(self.encoder_source_61(s))
        s = F.relu(self.encoder_source_62(s))

        # layer7
        p = F.relu(self.encoder_payload_7(p))
        s = F.relu(self.encoder_source_7(s))

        # layer8
        p = F.relu(self.encoder_payload_8(p))
        s4 = torch.cat((s, p, s3), 1)
        s = F.relu(self.encoder_source_8(s4))
        s = F.relu(self.encoder_source_81(s))
        s = F.relu(self.encoder_source_82(s))

        # layer9
        s = F.relu(self.encoder_source_9(s))

        # layer10
        s = F.relu(self.encoder_source_10(s))

        # layer11
        encoder_output = self.encoder_source_11(s)

        return encoder_output

    # TODO esto es el ejercicio 4
    def predict_decoder(self, encoded_image):
        # -------------------- Decoder --------------------------

        d = encoded_image.view(-1, 3, SIZE_IMAGE, SIZE_IMAGE)

        # layer1
        d = F.relu(self.decoder_layers1(d))
        d = F.relu(self.decoder_layers2(d))

        # layer3
        d = F.relu(self.decoder_layers3(d))
        d = F.relu(self.decoder_layers4(d))

        init_d = F.relu(self.decoder_layers5(d))

        # ---------------- decoder_payload ----------------

        # layer 1 & 2
        d = F.relu(self.decoder_payload1(init_d))
        d = F.relu(self.decoder_payload2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_payload3(d))
        d = F.relu(self.decoder_payload4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_payload5(d))
        decoded_payload = self.decoder_payload6(d)

        return decoded_payload
