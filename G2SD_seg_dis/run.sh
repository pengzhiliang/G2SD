############distill##################################
bash src/tools/dist_distill.sh \
    src/configs/distill/distill_cwd.py 1 \
    --work-dir output --seed 0  --deterministic \
    --options distiller.teacher_pretrained=teacher.pth \
              weight=3. \
              tau=1 \
              distiller.student_pretrained=student.pth \
         


############test##################################
bash src/tools/dist_test_distll.sh \
    src/configs/distill/distill_cwd.py \
     checkpoint.pth \
    1  --eval mIoU \
     --options distiller.teacher_pretrained=teacher.pth \
               distiller.student_pretrained=student.pth \

